import Foundation

/// Read-only GGUF v3 reader. Tensor payloads stay mapped until copied to MLX.
final class GGUFFile {
    struct Failure: LocalizedError {
        let message: String
        var errorDescription: String? { "GGUF: \(message)" }
    }
    struct Tensor {
        let name: String
        let dimensions: [Int] // GGML order: contiguous dimension first.
        let type: UInt32
        let offset: Int
        let byteCount: Int
        var shape: [Int] { dimensions.reversed() }
    }
    let data: Data
    private(set) var metadata: [String: Any] = [:]
    private(set) var tensors: [String: Tensor] = [:]
    private var cursor = 0

    static func layout(_ type: UInt32) throws -> (block: Int, bytes: Int, symbol: String) {
        switch type {
        case 0: return (1, 4, "f32")
        case 1: return (1, 2, "f16")
        case 8: return (32, 34, "q8_0")
        case 10: return (256, 84, "q2_K")
        case 12: return (256, 144, "q4_K")
        case 13: return (256, 176, "q5_K")
        case 14: return (256, 210, "q6_K")
        case 18: return (256, 98, "iq3_xxs")
        case 20: return (32, 18, "iq4_nl")
        case 21: return (256, 110, "iq3_s")
        case 22: return (256, 82, "iq2_s")
        case 23: return (256, 136, "iq4_xs")
        case 30: return (1, 2, "bf16")
        case 42: return (64, 18, "q2_0")
        default: throw Failure(message: "Unsupported quantization type \(type)")
        }
    }

    init(url: URL) throws {
        data = try Data(contentsOf: url, options: .alwaysMapped)
        guard try integer(UInt32.self) == 0x46554747, try integer(UInt32.self) == 3 else {
            throw Failure(message: "Expected little-endian GGUF v3")
        }
        let tensorCount = try count(limit: 100_000), metadataCount = try count(limit: 100_000)
        for _ in 0..<metadataCount {
            let key = try string()
            guard metadata[key] == nil else { throw Failure(message: "Duplicate metadata \(key)") }
            metadata[key] = try value(type: integer(UInt32.self), depth: 0)
        }
        var descriptions: [(String, [Int], UInt32, Int)] = []
        for _ in 0..<tensorCount {
            let name = try string(), rank = Int(try integer(UInt32.self))
            guard (1...4).contains(rank) else { throw Failure(message: "Invalid tensor rank") }
            let dimensions = try (0..<rank).map { _ in try count(limit: Int(Int32.max)) }
            guard dimensions.allSatisfy({ $0 > 0 }) else { throw Failure(message: "Empty tensor") }
            let type = try integer(UInt32.self), offset = try count(limit: data.count)
            descriptions.append((name, dimensions, type, offset))
        }
        let alignment = (metadata["general.alignment"] as? UInt32).map(Int.init) ?? 32
        guard alignment > 0, alignment <= 4096, alignment.nonzeroBitCount == 1 else {
            throw Failure(message: "Invalid alignment")
        }
        let base = (cursor + alignment - 1) / alignment * alignment
        var ranges: [Range<Int>] = []
        for (name, dimensions, type, relative) in descriptions {
            let layout = try Self.layout(type)
            guard dimensions[0] % layout.block == 0 else { throw Failure(message: "Unaligned quantized row") }
            var bytes = dimensions[0] / layout.block * layout.bytes
            for dim in dimensions.dropFirst() {
                let result = bytes.multipliedReportingOverflow(by: dim)
                guard !result.overflow else { throw Failure(message: "Tensor size overflow") }
                bytes = result.partialValue
            }
            guard base <= data.count, relative <= data.count - base,
                  bytes <= data.count - base - relative, tensors[name] == nil else {
                throw Failure(message: "Truncated or duplicate tensor \(name)")
            }
            let offset = base + relative
            tensors[name] = Tensor(name: name, dimensions: dimensions, type: type, offset: offset, byteCount: bytes)
            ranges.append(offset..<(offset + bytes))
        }
        ranges.sort { $0.lowerBound < $1.lowerBound }
        for index in 1..<max(1, ranges.count) where ranges[index-1].upperBound > ranges[index].lowerBound {
            throw Failure(message: "Overlapping tensors")
        }
    }

    private func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        guard MemoryLayout<T>.size <= data.count - cursor else { throw Failure(message: "Truncated header") }
        defer { cursor += MemoryLayout<T>.size }
        return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: T.self)) }
    }
    private func count(limit: Int) throws -> Int {
        let n = try integer(UInt64.self)
        guard n <= UInt64(limit) else { throw Failure(message: "Invalid length") }
        return Int(n)
    }
    private func string() throws -> String {
        let length = try count(limit: min(16 * 1024 * 1024, data.count - cursor))
        guard length <= data.count - cursor,
              let result = String(data: data[cursor..<(cursor + length)], encoding: .utf8) else {
            throw Failure(message: "Invalid UTF-8 string")
        }
        cursor += length
        return result
    }
    private func value(type: UInt32, depth: Int) throws -> Any {
        switch type {
        case 0: return try integer(UInt8.self)
        case 1: return try integer(Int8.self)
        case 2: return try integer(UInt16.self)
        case 3: return try integer(Int16.self)
        case 4: return try integer(UInt32.self)
        case 5: return try integer(Int32.self)
        case 6: return Float(bitPattern: try integer(UInt32.self))
        case 7: return try integer(UInt8.self) != 0
        case 8: return try string()
        case 9:
            guard depth < 2 else { throw Failure(message: "Nested metadata arrays") }
            let subtype = try integer(UInt32.self), length = try count(limit: 2_000_000)
            return try (0..<length).map { _ in try value(type: subtype, depth: depth + 1) }
        case 10: return try integer(UInt64.self)
        case 11: return try integer(Int64.self)
        case 12: return Double(bitPattern: try integer(UInt64.self))
        default: throw Failure(message: "Unknown metadata type \(type)")
        }
    }
}
