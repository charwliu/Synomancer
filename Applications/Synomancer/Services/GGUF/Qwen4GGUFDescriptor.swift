import Foundation

struct Qwen4GGUFDescriptor: Codable {
    let firstShard: String
    let projector: String?
    var mtp: String? = nil

    static func read(_ directory: URL) throws -> Self {
        let descriptor = try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("synomancer-gguf.json")))
        for path in [descriptor.firstShard, descriptor.projector, descriptor.mtp].compactMap({ $0 }) {
            guard path.hasSuffix(".gguf"), !path.contains("\\"), !path.contains("\0"),
                  path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
                throw GGUFFile.Failure(message: "Invalid relative asset path")
            }
        }
        guard descriptor.firstShard.contains("00001-of-00002") else {
            throw GGUFFile.Failure(message: "Expected two-shard Qwen4 checkpoint")
        }
        return descriptor
    }
    var shards: [String] { [firstShard, firstShard.replacingOccurrences(of: "00001-of-00002", with: "00002-of-00002")] }
}
