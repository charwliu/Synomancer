//
//  Logging.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//
import Synogging

public enum SynoLogging: LoggingSystem {
    public enum Category: String, CaseIterable {
        case library
        case service
        case data
        case view
        case application
    }
}

internal protocol Loggable: Synogging.Loggable where Self.LoggingSystemType == SynoLogging {}


let logger = SynoLogging.logger(forCategory: .view)

