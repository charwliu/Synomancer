//
//  LLMEvaluator.swift
//  Synomancer
//
//  Created by Wei LIU on 2/17/25.
//

@preconcurrency import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import SwiftUI

@Observable
@MainActor
class LLMEvaluator {
    var running = false
    
    var output = ""
    var modelInfo = ""
    var stat = ""
    
    let modelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit
    
    let generateParameters = MLXLMCommon.GenerateParameters(temperature: 0.6)
    
    let maxTokens = 800
    
    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4
    
    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }
    
    var loadState: LoadState = .idle
    
    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            let cache = try await SpeechCacheAccess.authorizedConfiguration()
            let models = try await LocalChatService.shared.models(cache: cache)
            let id = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
            guard let model = models.first(where: { $0.id == id }) else {
                throw LocalChatService.ChatError.modelUnavailable(id)
            }
            let root = try cache.resolve()
            let accessing = root.startAccessingSecurityScopedResource()
            defer { if accessing { root.stopAccessingSecurityScopedResource() } }
            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: model.directory, using: LocalChatTokenizerLoader())
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }
            
            self.modelInfo = "Loaded \(modelConfiguration.id). Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer
            
        case .loaded(let modelContainer):
            return modelContainer
        }
    }
    
    func generate(prompt: String, image: CIImage?, videoURL: URL?) async {
        guard !running else { return }
        
        running = true
        self.output = ""
        
        do {
            let modelContainer = try await load()
            
            let parameters = GenerateParameters(maxTokens: maxTokens, temperature: generateParameters.temperature)
            let cadence = displayEveryNTokens
            let result = try await modelContainer.perform { context in
                let images: [UserInput.Image] =
                if let image {
                    [UserInput.Image.ciImage(image)]
                } else {
                    []
                }
                let videos: [UserInput.Video] =
                if let videoURL {
                    [.url(videoURL)]
                } else {
                    []
                }
                var userInput = UserInput(
                    messages: [
                        [
                            "role": "user",
                            "content": [
                                ["type": "text", "text": prompt]
                            ]
                            + images.map { _ in
                                ["type": "image"]
                            }
                            + videos.map { _ in
                                ["type": "video"]
                            },
                        ]
                    ], images: images, videos: videos)
                userInput.processing.resize = .init(width: 448, height: 448)
                
                let input = try await context.processor.prepare(input: userInput)
                
                let (events, worker) = try MLXLMCommon.generateTokensTask(
                    input: input, parameters: parameters, context: context)
                return try await withTaskCancellationHandler {
                    var tokens: [Int] = []
                    var info: GenerateCompletionInfo?
                    do {
                        for await event in events {
                            try Task.checkCancellation()
                            switch event {
                            case .token(let token):
                                tokens.append(token)
                                if tokens.count % cadence == 0 {
                                    let text = context.tokenizer.decode(tokenIds: tokens)
                                    Task { @MainActor in self.output = text }
                                }
                            case .info(let completion): info = completion
                            }
                        }
                        await worker.value
                        try Task.checkCancellation()
                        return LocalChatResult(text: context.tokenizer.decode(tokenIds: tokens),
                                               tokensPerSecond: info?.tokensPerSecond ?? 0)
                    } catch {
                        worker.cancel()
                        await worker.value
                        throw error
                    }
                } onCancel: { worker.cancel() }
            }
            
            // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
            if result.text != self.output {
                self.output = result.text
            }
            self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"
            
        } catch {
            output = "Failed: \(error)"
        }
        
        running = false
    }
}

