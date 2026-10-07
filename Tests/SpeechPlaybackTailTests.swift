import Foundation
import AVFoundation

@main
struct SpeechPlaybackTailTests {
    static func render(_ input: [Float], sizes: [Int]) -> [Float] {
        var tail = SpeechPlaybackTail()
        var output: [Float] = []
        var offset = 0, index = 0
        while offset < input.count {
            let end = min(input.count, offset + sizes[index % sizes.count])
            output += tail.append(Array(input[offset..<end]))
            // An empty codec EOS/metadata chunk must not discard the held tail.
            precondition(tail.append([]).isEmpty)
            offset = end
            index += 1
        }
        output += tail.finish()
        precondition(tail.finish().isEmpty, "EOS duplicated the tail")
        return output
    }

    static func check(_ input: [Float], output: [Float]) {
        let fade = min(480, input.count)
        precondition(output.count == input.count + 1200)
        precondition(output.prefix(input.count - fade).elementsEqual(input.prefix(input.count - fade)),
                     "Intermediate chunk/segment samples changed")
        precondition(output.suffix(1201).allSatisfy { $0 == 0 }, "Output device did not finish on silence")
        for index in (input.count - fade)..<input.count {
            precondition(abs(output[index]) <= abs(input[index]) + 1e-7)
        }
    }

    static func main() throws {
        let constant = Array(repeating: Float(0.8), count: 4097)
        let result = render(constant, sizes: [1, 17, 960, 37])
        check(constant, output: result)
        precondition(result[constant.count - 480] == constant[constant.count - 480])
        let ramp = result[(constant.count - 480)..<constant.count]
        precondition(zip(ramp, ramp.dropFirst()).allSatisfy { $0 >= $1 })
        for count in [1, 17, 479, 480, 481] {
            let input = Array(repeating: Float(-0.3), count: count)
            check(input, output: render(input, sizes: [3, 480]))
        }
        var empty = SpeechPlaybackTail()
        precondition(empty.append([]).isEmpty && empty.finish().isEmpty)
        let sine = (0..<10337).map { Float(sin(Double($0) * 0.097) * 0.6) }
        let expected = render(sine, sizes: [sine.count])
        for sizes in [[1], [4320, 23040], [13, 2048, 3, 481]] {
            let actual = render(sine, sizes: sizes)
            check(sine, output: actual)
            precondition(actual == expected, "Fade depends on the streaming split")
        }
        print("PASS: one EOS fade, exact intermediate PCM, tiny/empty streams, empty EOS sentinels and chunk-independent sample count")

        if CommandLine.arguments.count > 1 {
            let directory = URL(fileURLWithPath: CommandLine.arguments[1])
            for index in 0...1 {
                let file = try AVAudioFile(forReading: directory.appendingPathComponent("tts-stream-\(index).wav"))
                precondition(file.processingFormat.sampleRate == 24000 && file.processingFormat.channelCount == 1)
                let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
                try file.read(into: buffer)
                let input = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
                let output = render(input, sizes: [4320, 23040])
                check(input, output: output)
                let path = directory.appendingPathComponent("tts-tail-fixed-\(index).wav")
                let writer = try AVAudioFile(forWriting: path, settings: file.fileFormat.settings,
                                            commonFormat: .pcmFormatFloat32, interleaved: false)
                let processed = AVAudioPCMBuffer(pcmFormat: writer.processingFormat, frameCapacity: AVAudioFrameCount(output.count))!
                processed.frameLength = processed.frameCapacity
                output.withUnsafeBufferPointer { processed.floatChannelData![0].update(from: $0.baseAddress!, count: $0.count) }
                try writer.write(from: processed)
                print("PASS: real TTS fixture \(index) hard-stop amplitude \(input.last!) → 0; all but final 20ms unchanged; added silence=50ms")
            }
        }
    }
}
