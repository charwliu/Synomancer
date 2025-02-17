//
//  Flag.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//


import Foundation

/// A Flag represents the state of a `Transaction`
public enum Flag: String {

    /// complete to mark that the transaction was checked
    case complete = "*"

    /// incomplete to mark that the transaction requires further attantion
    case incomplete = "!"
}

extension Flag: CustomStringConvertible {

    /// Retuns the `String` which represents the flag in the ledger file
    public var description: String { self.rawValue }

}