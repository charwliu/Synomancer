//
//  PluginParser.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


import Foundation
import BeanCountParserUtils

enum PluginParser {

    private static let regex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: "^plugin\\s+\"([^\"]*)\"\\s*(;.*)?$", options: [])
    }()

    static func parseFrom(line: String) -> String? {
        let matches = line.matchingStrings(regex: self.regex)
        guard let match = matches[safe: 0] else {
            return nil
        }
        return match[1]
    }

}
