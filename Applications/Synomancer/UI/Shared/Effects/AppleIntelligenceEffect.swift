// Adapted from ChatMLX AppleIntelligenceEffectView, created by John Mai on 2024/10/6.
// Apache-2.0; see docs/AppleIntelligenceEffect.md and docs/licenses/ChatMLX-Apache-2.0.txt.
// Synomancer: view-local lifecycle, bounded refresh rate and accessibility support.
import SwiftUI

private struct AppleIntelligenceEffect: ViewModifier {
    @AppStorage("appleIntelligenceEffectEnabled") private var enabled = true
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.scenePhase) private var scenePhase
    let active: Bool

    private var visible: Bool { enabled && active && scenePhase == .active }

    func body(content: Content) -> some View {
        content.overlay {
            if visible {
                AppleIntelligenceEffectView(animated: !reduceMotion)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.35), value: visible)
    }
}

struct AppleIntelligenceEffectView: View {
    var animated = true
    @State private var started = Date()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !animated)) { timeline in
            let time = animated ? timeline.date.timeIntervalSince(started) : 0
            GeometryReader { geometry in
                ZStack {
                    ForEach(0..<2) { index in
                        let blur = index == 0 ? 18 + 6 * sin(time * 2) : 6 + 3 * sin(time * 4)
                        Rectangle()
                            .fill(ShaderLibrary.colorWheel(.boundingRect, .float(1)))
                            // Cover wide and tall windows throughout rotation.
                            .frame(width: hypot(geometry.size.width, geometry.size.height),
                                   height: hypot(geometry.size.width, geometry.size.height))
                            .rotationEffect(.degrees(time * (index == 0 ? 193 : -73)))
                            .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                            .mask {
                                RoundedRectangle(cornerRadius: 18)
                                    .strokeBorder(lineWidth: 8)
                                    .blur(radius: blur)
                            }
                    }
                }
            }
            .clipped()
        }
    }
}

extension View {
    func appleIntelligenceEffect(active: Bool) -> some View {
        modifier(AppleIntelligenceEffect(active: active))
    }
}

#Preview("Generating") {
    Text("思考中…")
        .frame(width: 640, height: 420)
        .background(.background)
        .overlay { AppleIntelligenceEffectView() }
}
