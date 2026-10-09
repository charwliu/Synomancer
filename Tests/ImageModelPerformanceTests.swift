import Foundation
import MLX
import Synchronization

/// Actual model benchmark, using the production pipelines and their default steps.
/// First means first request of a service session, not an OS-cold disk cache.
enum ImageModelPerformanceTests {
    struct Event: Sendable { let stage: String; let time: Double }

    static func run(cache: SpeechCacheAccess.Configuration, output: URL, size: Int) async throws {
        let discovery = LocalImageService()
        let models = try await discovery.models(cache: cache)
        let ids = ["black-forest-labs/FLUX.1-dev", "Qwen/Qwen-Image-2.1",
                   "mlx-community/Qwen-Image-2.1-MLX-4bit"]
        let selected = try ids.map { id -> LocalImageModel in
            guard let model = models.first(where: { $0.id == id }) else {
                throw LocalImageError.missing(id)
            }
            return model
        }
        let services = selected.map { _ in LocalImageService() }
        let prompt = "Product photograph of a glossy red ceramic mug with a handle on the right, on a white table against a soft light gray background, soft studio lighting, centered composition, realistic, no text."
        let clock = ContinuousClock()
        let origin = clock.now
        func elapsed() -> Double {
            let d = origin.duration(to: clock.now).components
            return Double(d.seconds) + Double(d.attoseconds) / 1e18
        }
        var rows: [[String: Any]] = []
        var firstImages: [Int: Data] = [:]
        // Balanced order; each service retains only its own conditioning/prefix.
        for (trial, modelIndex) in [0, 1, 2, 2, 1, 0].enumerated() {
            let model = selected[modelIndex]
            Memory.clearCache()
            Memory.peakMemory = 0
            let beforeActive = Memory.activeMemory
            let events = Mutex<[Event]>([])
            let start = elapsed()
            let request = LocalImageRequest(modelID: model.id, prompt: prompt,
                width: size, height: size, steps: model.family.defaultSteps, seed: 42, guidance: 3.5)
            let repeated = firstImages[modelIndex] != nil
            print("BENCH START trial=\(trial) model=\(model.id) repeat=\(repeated) size=\(size) steps=\(request.steps)")
            let result = try await services[modelIndex].generate(request, cache: cache) { stage, _ in
                let d = origin.duration(to: clock.now).components
                let time = Double(d.seconds) + Double(d.attoseconds) / 1e18
                events.withLock { $0.append(Event(stage: stage, time: time)) }
                print("EVENT \(time - start) \(stage)")
            }
            let end = elapsed()
            try LocalImageSmoke.checkImage(result.png, size: size)
            if let first = firstImages[modelIndex] {
                try LocalImageSmoke.compareImage(result.png, first, label: "model benchmark repeated \(model.id)")
            } else { firstImages[modelIndex] = result.png }
            let name = "\(modelIndex)-\(repeated ? "repeat" : "first")-\(size).png"
            try result.png.write(to: output.appendingPathComponent(name))
            let stages = events.withLock { $0 }
            let steps = stages.filter { $0.stage.hasPrefix("生成图像 ") || $0.stage.hasPrefix("FLUX.1 去噪 ") }
            guard steps.count == request.steps,
                  let decode = stages.first(where: { $0.stage.hasPrefix("解码") }) else {
                throw LocalImageError.invalid("Missing benchmark phase markers")
            }
            let row: [String: Any] = [
                "trial": trial, "model": model.id, "repeat": repeated,
                "width": size, "height": size, "steps": request.steps, "seed": request.seed,
                "guidance": request.guidance, "componentGB": Double(model.bytes) / 1e9,
                "wallSeconds": end - start, "pipelineSeconds": result.seconds,
                "loadEncodeSeconds": steps[0].time - start,
                "denoiseSeconds": decode.time - steps[0].time,
                "firstStepSeconds": steps[1].time - steps[0].time,
                "steadyStepSeconds": (decode.time - steps[1].time) / Double(request.steps - 1),
                "decodePNGSeconds": end - decode.time,
                "peakMLXGB": Double(Memory.peakMemory) / 1e9,
                "activeBeforeGB": Double(beforeActive) / 1e9,
                "reuseStages": stages.map(\.stage).filter { $0.hasPrefix("复用") },
                "png": name,
            ]
            rows.append(row)
            let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
            print("BENCH RESULT \(String(decoding: data, as: UTF8.self))")
            try JSONSerialization.data(withJSONObject: ["prompt": prompt, "runs": rows], options: [.prettyPrinted, .sortedKeys])
                .write(to: output.appendingPathComponent("results-\(size).json"))
        }
        for service in services { await service.clearSessionCache() }
        print("PASS: six actual image requests, phase timing, PNG checks and repeated-output tolerance")
    }
}
