//
//  Loggable.swift
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

/// Loggable type for a ``LoggingSystem``.
public protocol Loggable<LoggingSystemType>: Syno, Mancer
    where LoggingSystemType: LoggingSystem {
}
