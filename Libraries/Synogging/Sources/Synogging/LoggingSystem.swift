//
//  LoggingSystem.swift
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

// swiftlint:disable strict_fileprivate
private class LoggingSystemRepository: @unchecked Sendable {
    fileprivate static let shared = LoggingSystemRepository()
    
    private let lock = NSRecursiveLock()
    private var items = [String: Any]()
    
    private init(items: [String: Any] = [String: Any]()) {
        self.items = items
    }
    
    fileprivate func loggingSystem<LoggingSystemType: LoggingSystem>(
        for system: LoggingSystemType.Type,
        using value: @autoclosure () -> [LoggingSystemType.Category: Logger]
    ) -> [LoggingSystemType.Category: Logger] {
        let anyItem = lock.withLock {
            items[system.identifier]
        }
        if let item = anyItem as? [LoggingSystemType.Category: Logger] {
            return item
        } else {
            assert(anyItem == nil)
            return lock.withLock {
                let value = value()
                items[system.identifier] = value
                return value
            }
        }
    }
}

// swiftlint:enable strict_fileprivate

/// Defines the logging categories for your application.
public protocol LoggingSystem {
    /// Logging categories available to types in the application
    associatedtype Category: Hashable & RawRepresentable
    where Category.RawValue == String
    
    // swiftlint:disable:next missing_docs
    static var identifier: String { get }
    
    /// Subsystem to use for each ``Logger``.
    /// By default, this is `Bundle.main.bundleIdentifier`.
    static var subsystem: String { get }
    
    /// Fetches the correct logger based on the category.
    static func logger(forCategory category: Category) -> Logger
}

extension LoggingSystem {
    // swiftlint:disable:next missing_docs
    public static var identifier: String {
        String(reflecting: Self.self)
    }
    
    /// By default, this is `Bundle.main.bundleIdentifier`.
    public static var subsystem: String {
#if canImport(os)
        // swiftlint:disable:next force_unwrapping
        Bundle.main.bundleIdentifier!
#else
        identifier
#endif
    }
}

extension LoggingSystem where Category: CaseIterable {
    private static var loggers: [Category: Logger] {
        LoggingSystemRepository.shared.loggingSystem(for: Self.self, using: defaultLoggers())
    }
    
    /// If ``Category`` implements `CaseIterable`, ``LoggingSystem`` can automatically
    /// iterate over the cases and automatically create the ``Logger`` objects needed.
    public static func logger(forCategory category: Category) -> Logger {
        guard let logger = loggers[category] else {
            preconditionFailure("missing logger")
        }
        return logger
    }
    
    private static func defaultLoggers() -> [Category: Logger] {
        .init(
            uniqueKeysWithValues: Category.allCases.map {
                ($0, Logger(subsystem: Self.subsystem, category: $0))
            }
        )
    }
}

