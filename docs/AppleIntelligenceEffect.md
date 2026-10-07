# Apple Intelligence-style activity effect

Adapted from John Mai's ChatMLX component:
https://github.com/johnmai-dev/ChatMLX/tree/main/ChatMLX/Components/AppleIntelligenceEffect

Upstream source blobs inspected:
- AppleIntelligenceEffectView.swift: 54386585fe28f315c1cda31582641743f4cd786f
- ColorWheel.metal: 4e6afca4fdce9e88be9ff7e21c32edcc58efde34

The original SwiftUI view and Metal color wheel were created by John Mai on
2024-10-06. The upstream repository uses Apache-2.0; a copy is included at
`licenses/ChatMLX-Apache-2.0.txt`. Original shader attribution is preserved.

Synomancer adapts the two rotating color fields and sinusoidal blur into a
view-local overlay. It covers the chat content while the conversation is loading
and the image studio while generation, control extraction, or upscaling is busy.
Completion, cancellation and errors use existing store state to remove the effect.
No separate screen-sized NSWindow or singleton window controller is required.
The overlay does not intercept input or appear in the accessibility tree.

Settings > 视觉效果 > 生成时显示彩色流光 controls the effect (default on).
Animation refresh is limited to 30 Hz; Reduce Motion uses a static border.
Inactive scenes remove the overlay. Transitions last 0.35 seconds and are disabled
with Reduce Motion. No new package dependencies or Apple Intelligence services.

This visual component uses the GPU; its inference performance impact has not been
benchmarked. Disable it in Settings when collecting inference timing baselines.
