//
//  Option.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//


import Foundation

/// Option to control bahaviour of Beancount
public struct Option: Sendable {

    /// name of the option
    public let name: String
    /// value the option should be set to
    public let value: String

    /// Initializes a Option
    /// - Parameters:
    ///   - name: name of the option
    ///   - value: value the option should be set to
    public init(name: String, value: String) {
        self.name = name
        self.value = value
    }

}

extension Option: CustomStringConvertible {

    public var description: String {
        "option \"\(name)\" \"\(value)\""
    }

}

extension Option: Comparable {

    public static func < (lhs: Option, rhs: Option) -> Bool {
        String(describing: lhs) < String(describing: rhs)
    }

}
