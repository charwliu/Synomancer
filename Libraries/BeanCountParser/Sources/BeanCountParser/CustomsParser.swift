//
//  CustomsParser.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


import Foundation
import BeanCountModel
import BeanCountParserUtils

enum CustomsParser {

    private static let regex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: "^\(DateParser.dateGroup)\\s+custom\\s+\"([^\"]*)\"((\\s+\"([^\"]*)\")+)\\s*(;.*)?$", options: [])
    }()

    static func parseFrom(line: String, metaData: [String: String] = [:]) -> Custom? {
        let matches = line.matchingStrings(regex: self.regex)
        guard let match = matches[safe: 0], let date = DateParser.parseFrom(string: match[1]) else {
            return nil
        }
        let values: [String] = match[3].split(separator: "\"").compactMap {
            let trimmed = $0.trimmingCharacters(in: .whitespaces)
            return trimmed.isEmpty ? nil : String($0)
        }
        return Custom(date: date, name: match[2], values: values, metaData: metaData)
    }

}
