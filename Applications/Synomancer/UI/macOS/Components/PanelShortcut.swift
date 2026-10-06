#if os(macOS)
import AppKit
import Carbon
import SwiftUI

@Observable @MainActor
final class PanelShortcut {
    static let shared = PanelShortcut()
    private(set) var label: String
    private(set) var error: String?
    private var keyCode: UInt32
    private var modifiers: UInt32
    private var hotKey: EventHotKeyRef?
    private var handler: EventHandlerRef?
    private var action: (() -> Void)?

    private init() {
        let defaults = UserDefaults.standard
        keyCode = defaults.object(forKey: "panelShortcut.keyCode") as? UInt32 ?? UInt32(kVK_ANSI_K)
        modifiers = defaults.object(forKey: "panelShortcut.modifiers") as? UInt32 ?? UInt32(cmdKey | optionKey)
        label = defaults.string(forKey: "panelShortcut.label") ?? "⌥⌘K"
    }

    func activate(_ action: @escaping () -> Void) {
        self.action = action
        guard handler == nil else { return }
        var event = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, _ in
            guard let event else { return OSStatus(eventNotHandledErr) }
            var identifier = EventHotKeyID()
            let status = GetEventParameter(
                event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                nil, MemoryLayout<EventHotKeyID>.size, nil, &identifier)
            guard status == noErr, identifier.signature == 0x53594E4F, identifier.id == 1 else {
                return OSStatus(eventNotHandledErr)
            }
            Task { @MainActor in PanelShortcut.shared.action?() }
            return noErr
        }, 1, &event, nil, &handler)
        guard status == noErr else {
            error = "Could not install keyboard shortcut (\(status))."
            return
        }
        register()
    }

    func beginRecording() {
        if let hotKey { UnregisterEventHotKey(hotKey) }
        hotKey = nil
    }

    func cancelRecording() {
        if hotKey == nil { register() }
    }

    func record(_ event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard !flags.intersection([.command, .option, .control]).isEmpty else {
            register()
            error = "Use Command, Option or Control with the shortcut."
            return
        }
        let previous = (keyCode, modifiers, label)
        if let hotKey { UnregisterEventHotKey(hotKey) }
        hotKey = nil
        keyCode = UInt32(event.keyCode)
        modifiers = 0
        if flags.contains(.command) { modifiers |= UInt32(cmdKey) }
        if flags.contains(.option) { modifiers |= UInt32(optionKey) }
        if flags.contains(.control) { modifiers |= UInt32(controlKey) }
        if flags.contains(.shift) { modifiers |= UInt32(shiftKey) }
        label = (flags.contains(.control) ? "⌃" : "") + (flags.contains(.option) ? "⌥" : "")
            + (flags.contains(.shift) ? "⇧" : "") + (flags.contains(.command) ? "⌘" : "")
            + (event.charactersIgnoringModifiers?.uppercased() ?? "Key \(event.keyCode)")
        register()
        if error != nil {
            let failure = error
            (keyCode, modifiers, label) = previous
            register()
            error = failure
        } else {
            UserDefaults.standard.set(keyCode, forKey: "panelShortcut.keyCode")
            UserDefaults.standard.set(modifiers, forKey: "panelShortcut.modifiers")
            UserDefaults.standard.set(label, forKey: "panelShortcut.label")
        }
    }

    private func register() {
        let status = RegisterEventHotKey(
            keyCode, modifiers, EventHotKeyID(signature: 0x53594E4F, id: 1),
            GetApplicationEventTarget(), 0, &hotKey)
        error = status == noErr ? nil : "Keyboard shortcut is unavailable (\(status))."
    }
}

struct PanelShortcutRecorder: View {
    @State private var recording = false
    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("Keyboard shortcut")
                Button(recording ? "Press a shortcut…" : PanelShortcut.shared.label) {
                    PanelShortcut.shared.beginRecording()
                    recording = true
                }
                    .background(ShortcutKeyReader(recording: $recording).frame(width: 0, height: 0))
            }
            if let error = PanelShortcut.shared.error { Text(error).font(.caption).foregroundStyle(.red) }
        }
        .onDisappear {
            if recording { PanelShortcut.shared.cancelRecording() }
        }
    }
}

private struct ShortcutKeyReader: NSViewRepresentable {
    @Binding var recording: Bool
    func makeNSView(context: Context) -> KeyView { KeyView() }
    func updateNSView(_ view: KeyView, context: Context) {
        view.onKey = { [weak view] event in
            if event.keyCode == UInt16(kVK_Escape) {
                PanelShortcut.shared.cancelRecording()
            } else {
                PanelShortcut.shared.record(event)
            }
            recording = false
            view?.window?.makeFirstResponder(nil)
        }
        if recording { view.window?.makeFirstResponder(view) }
    }
    final class KeyView: NSView {
        var onKey: ((NSEvent) -> Void)?
        override var acceptsFirstResponder: Bool { true }
        override func keyDown(with event: NSEvent) { onKey?(event) }
        override func performKeyEquivalent(with event: NSEvent) -> Bool {
            guard window?.firstResponder === self else { return false }
            onKey?(event)
            return true
        }
    }
}
#endif
