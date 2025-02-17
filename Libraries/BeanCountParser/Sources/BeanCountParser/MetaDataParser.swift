//
//  MetaDataParser.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


import Foundation
import BeanCountParserUtils

enum MetaDataParser {

    static let metaDataGroup = ""

    private static let regex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: "^  (  )?([^\\s]+):\\s*\"([^\"]+)\"\\s*(;.*)?$", options: [])
    }()

    /// Parse MetaData from a line String
    ///
    /// - Parameters:
    ///   - line: string of one line
    /// - Returns: a [String: String] or nil if the line does not contain valid meta data
    static func parseFrom(line: String) -> [String: String]? { // swiftlint:disable:this discouraged_optional_collection
        let matches = line.matchingStrings(regex: self.regex)
        guard let match = matches[safe: 0] else {
            return nil
        }
        return [match[2]: match[3]]
    }

}
