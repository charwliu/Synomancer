//
//  Clipboard.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//



import Foundation

#if os(macOS)
import AppKit
#else
import UIKit
#endif


final class Clipboard: Sendable {
    static let shared = Clipboard()
    
    func setString(_ message: String) {
#if os(iOS)
        UIPasteboard.general.string = message
#elseif os(macOS)
        let pasteboard = NSPasteboard.general
        pasteboard.declareTypes([.string], owner: nil)
        pasteboard.setString(message, forType: .string)
#endif
    }
    
    func getImage() -> PlatformImage? {
#if os(iOS)
        if let image = UIPasteboard.general.image {
            return image
        }
        return nil
#elseif os(macOS)
        let pb = NSPasteboard.general
        let type = NSPasteboard.PasteboardType.tiff
        guard let imgData = pb.data(forType: type) else { return nil }
        return NSImage(data: imgData)
#endif
    }
    
    func getText() -> String? {
#if os(iOS) || os(visionOS)
        return UIPasteboard.general.string
#elseif os(macOS)
        return NSPasteboard.general.string(forType: .string)
#endif
    }
}


