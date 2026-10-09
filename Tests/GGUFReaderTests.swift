import Foundation

@main
struct GGUFReaderTests {
    static func main() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.gguf")
        func u32(_ n: UInt32) -> Data { var v = n.littleEndian; return withUnsafeBytes(of: &v) { Data($0) } }
        func u64(_ n: UInt64) -> Data { var v = n.littleEndian; return withUnsafeBytes(of: &v) { Data($0) } }
        func make(type: UInt32 = 0, offset: UInt64 = 0) -> Data {
            var data = u32(0x46554747) + u32(3) + u64(1) + u64(0)
            data += u64(6) + Data("weight".utf8) + u32(1) + u64(4) + u32(type) + u64(offset)
            while data.count % 32 != 0 { data.append(0) }
            data += u32(0x3f800000) + u32(0x40000000) + u32(0x40400000) + u32(0x40800000)
            return data
        }
        let good = make()
        try good.write(to: url)
        let file = try GGUFFile(url: url)
        precondition(file.tensors["weight"]?.dimensions == [4])
        precondition(file.tensors["weight"]?.byteCount == 16)
        for bad in [Data(), Data(good.prefix(12)), Data(good.dropLast()), make(type: 999), make(offset: UInt64.max)] {
            try bad.write(to: url)
            do { _ = try GGUFFile(url: url); fatalError("Accepted invalid GGUF") }
            catch is GGUFFile.Failure {}
        }
        for path in ["/tmp/a.gguf", "../a.gguf", "a/../b.gguf", "a\\b.gguf", "a//b.gguf", "single.gguf"] {
            let descriptor = Qwen4GGUFDescriptor(firstShard: path, projector: nil)
            try JSONEncoder().encode(descriptor).write(to: directory.appendingPathComponent("synomancer-gguf.json"))
            do { _ = try Qwen4GGUFDescriptor.read(directory); fatalError("Accepted invalid asset path") }
            catch is GGUFFile.Failure {}
        }
        for path in ["/tmp/draft.gguf", "../draft.gguf", "mtp/../draft.gguf", "mtp//draft.gguf"] {
            let descriptor = Qwen4GGUFDescriptor(firstShard: "IQ3_S/model-00001-of-00002.gguf", projector: nil, mtp: path)
            try JSONEncoder().encode(descriptor).write(to: directory.appendingPathComponent("synomancer-gguf.json"))
            do { _ = try Qwen4GGUFDescriptor.read(directory); fatalError("Accepted invalid MTP path") }
            catch is GGUFFile.Failure {}
        }
        let descriptor = Qwen4GGUFDescriptor(firstShard: "IQ3_S/model-00001-of-00002.gguf", projector: "vision/mmproj.gguf", mtp: "mtp/complete.gguf")
        try JSONEncoder().encode(descriptor).write(to: directory.appendingPathComponent("synomancer-gguf.json"))
        let loaded = try Qwen4GGUFDescriptor.read(directory)
        precondition(loaded.shards.count == 2 && loaded.mtp == "mtp/complete.gguf")
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"] {
            try Data("{}".utf8).write(to: directory.appendingPathComponent(name))
        }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("IQ3_S"), withIntermediateDirectories: true)
        for path in descriptor.shards { try good.write(to: directory.appendingPathComponent(path)) }
        var discovered = try LocalChatModelCache.scan(in: directory, llmTypes: [], vlmTypes: [])
        precondition(discovered.count == 1 && discovered[0].modelType == "qwen4_exp_gguf" && !discovered[0].supportsImages)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("vision"), withIntermediateDirectories: true)
        try good.write(to: directory.appendingPathComponent(descriptor.projector!))
        discovered = try LocalChatModelCache.scan(in: directory, llmTypes: [], vlmTypes: [])
        precondition(discovered[0].supportsImages)
        try FileManager.default.removeItem(at: directory.appendingPathComponent(descriptor.shards[1]))
        let incomplete = try LocalChatModelCache.scan(in: directory, llmTypes: [], vlmTypes: [])
        precondition(incomplete.isEmpty)
        print("PASS: GGUF bounds, quantization rejection, relative paths, split completeness and vision discovery")
    }
}
