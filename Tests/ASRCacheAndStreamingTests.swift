import Foundation
import AudioCommon

@main
struct ASRComponentsTests {
    static func main() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let repo = root.appendingPathComponent("hub/models--mlx-community--Qwen3-ASR-1.7B-bf16")
        let snapshot = repo.appendingPathComponent("snapshots/abcdef")
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try fm.createDirectory(at: repo.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try Data("abcdef".utf8).write(to: repo.appendingPathComponent("refs/main"))
        let config = "{\"model_type\":\"qwen3_asr\",\"thinker_config\":{\"text_config\":{\"hidden_size\":2048,\"num_hidden_layers\":28}}}"
        try Data(config.utf8).write(to: snapshot.appendingPathComponent("config.json"))
        for name in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
            try Data("{}".utf8).write(to: snapshot.appendingPathComponent(name))
        }
        let blob = repo.appendingPathComponent("blobs/weight")
        try fm.createDirectory(at: blob.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data([1]).write(to: blob)
        let weight = snapshot.appendingPathComponent("model.safetensors")
        try fm.createSymbolicLink(atPath: weight.path, withDestinationPath: "../../blobs/weight")
        for directory in [root, root.appendingPathComponent("hub"), repo, snapshot] {
            let selection = LocalASRCache.resolve(in: directory)
            precondition(selection?.directory.standardizedFileURL.path == snapshot.standardizedFileURL.path && selection?.modelId == "mlx-community/Qwen3-ASR-1.7B-bf16")
        }
        print("PASS: hub, cache root, repository, snapshot and symlink layouts")
        try fm.removeItem(at: snapshot.appendingPathComponent("vocab.json"))
        precondition(LocalASRCache.resolve(in: root) == nil)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("vocab.json"))
        try fm.removeItem(at: blob)
        precondition(LocalASRCache.resolve(in: root) == nil)
        try Data([1]).write(to: blob)
        let index = snapshot.appendingPathComponent("model.safetensors.index.json")
        try Data("{\"weight_map\":{\"a\":\"model.safetensors\",\"b\":\"missing.safetensors\"}}".utf8).write(to: index)
        precondition(LocalASRCache.resolve(in: root) == nil)
        print("PASS: missing tokenizer, dangling blob and incomplete shards rejected")
        try fm.removeItem(at: index)
        try Data(config.replacingOccurrences(of: "2048", with: "1024").utf8).write(to: snapshot.appendingPathComponent("config.json"))
        precondition(LocalASRCache.resolve(in: root) == nil)
        print("PASS: incompatible model rejected")

        let tts = root.appendingPathComponent("hub/models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16/snapshots/abc")
        try fm.createDirectory(at: tts.appendingPathComponent("speech_tokenizer"), withIntermediateDirectories: true)
        let ttsConfig = "{\"model_type\":\"qwen3_tts\",\"tts_model_type\":\"custom_voice\",\"talker_config\":{\"hidden_size\":2048}}"
        try Data(ttsConfig.utf8).write(to: tts.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: tts.appendingPathComponent("vocab.json"))
        try Data([1]).write(to: tts.appendingPathComponent("model.safetensors"))
        let codec = tts.appendingPathComponent("speech_tokenizer/model.safetensors")
        try Data([1]).write(to: codec)
        precondition(LocalTTSCache.resolve(in: root)?.model.resolvingSymlinksInPath().path == tts.resolvingSymlinksInPath().path)
        try fm.removeItem(at: codec)
        precondition(LocalTTSCache.resolve(in: root) == nil)
        try Data([1]).write(to: codec)
        try Data(ttsConfig.replacingOccurrences(of: "custom_voice", with: "base").utf8).write(to: tts.appendingPathComponent("config.json"))
        precondition(LocalTTSCache.resolve(in: root) == nil)
        print("PASS: TTS cache, missing codec and unsupported Base variant")

        let buffer = StreamingSpeechBuffer(sampleRate: 10, maximumSeconds: 3)
        buffer.append([Float](repeating: 0.5, count: 10))
        let first = buffer.snapshot()
        precondition(first.hasSpeech && !first.hasSilentTail)
        buffer.append([Float](repeating: 0, count: 10))
        precondition(first.samples == [Float](repeating: 0.5, count: 10))
        precondition(buffer.snapshot().hasSilentTail)
        buffer.discard(through: first.endFrame)
        precondition(buffer.snapshot().samples == [Float](repeating: 0, count: 10))
        precondition(!buffer.snapshot().hasSpeech)
        buffer.append([Float](repeating: 0.2, count: 40))
        precondition(buffer.snapshot().samples.count == 30)
        buffer.discard(through: first.endFrame)
        precondition(buffer.snapshot().samples.count == 30)
        print("PASS: immutable snapshots, silent-tail detection, concurrent arrivals and bounded storage")

        var transcript = StreamingSpeechTranscript()
        precondition(transcript.update(partial: "Can", commit: false) == "Can")
        precondition(transcript.update(partial: "Can you guarantee?", commit: true) == "Can you guarantee?")
        precondition(transcript.update(partial: "Tomorrow", commit: false) == "Can you guarantee? Tomorrow")
        precondition(transcript.update(partial: "Tomorrow morning.", commit: true) == "Can you guarantee? Tomorrow morning.")
        precondition(transcript.update(partial: "", commit: true) == "Can you guarantee? Tomorrow morning.")
        print("PASS: partial revisions, segment commits and empty final result")

        let concurrent = StreamingSpeechBuffer(sampleRate: 1000, maximumSeconds: 2)
        DispatchQueue.concurrentPerform(iterations: 1000) { _ in
            concurrent.append([0.25])
            _ = concurrent.snapshot()
        }
        precondition(concurrent.snapshot().samples.count == 1000)
        print("PASS: concurrent audio append and snapshot")
    }
}
