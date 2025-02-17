//
//  Logger.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/12.
//

// swiftlint:disable file_types_order

#if canImport(os)
import os
/// Standard os.log Logger
public typealias Logger = os.Logger
#else
import Logging
public typealias Logger = Logger.Logger
#endif

// swiftlint:enable file_types_order
