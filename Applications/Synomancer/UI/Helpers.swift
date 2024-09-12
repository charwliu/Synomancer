//
//  Helpers.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//


import SwiftUI

#if os(iOS) || os(visionOS)
typealias PlatformImage = UIImage
#else
typealias PlatformImage = NSImage
#endif

//Image(nsImage: nsImage)

