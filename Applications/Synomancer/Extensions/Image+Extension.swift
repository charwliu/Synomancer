//
//  Image+Extension.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import SwiftUI

#if os(iOS) || os(watchOS) || os(tvOS) || os(visionOS)
extension Image {
    init?(data: Data) {
        guard let uiImage = UIImage(data: data) else { return nil }
        self.init(uiImage: uiImage)
    }
}
#elseif os(macOS)
extension Image {
    init?(data: Data) {
        guard let nsImage = NSImage(data: data) else { return nil }
        self.init(nsImage: nsImage)
    }
}
#endif

