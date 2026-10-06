//
//  LLMEvaluator.swift
//  Synomancer
//
//  Created by Wei LIU on 2/17/25.
//

@preconcurrency import CoreImage
import MLX
import MLXLMCommon
import MLXRandom
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
            // limit the buffer cache
            MLX.Memory.cacheLimit = 20 * 1024 * 1024
            
            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) { [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                    "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            
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
            
            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
            
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
                
                return try MLXLMCommon.generate(
                    input: input,
                    parameters: generateParameters,
                    context: context
                ) { tokens in
                    // update the output -- this will make the view show the text as it generates
                    if tokens.count % displayEveryNTokens == 0 {
                        let text = context.tokenizer.decode(tokens: tokens)
                        Task { @MainActor in
                            self.output = text
                        }
                    }
                    
                    if tokens.count >= maxTokens {
                        return .stop
                    } else {
                        return .more
                    }
                }
            }
            
            // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
            if result.output != self.output {
                self.output = result.output
            }
            self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"
            
        } catch {
            output = "Failed: \(error)"
        }
        
        running = false
    }
}

