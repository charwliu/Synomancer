//
//  Mancer.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

import Foundation
#if canImport(os)
import os
#else
import Logging
#endif

/// Defines a shared logger for the type.
///
/// Provides a shared ``Logger`` to use in this type.
public protocol Mancer {
    /// Shared logger for Type.
    static var logger: Logger {
        get
    }
}

