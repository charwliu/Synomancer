//
//  EventParser.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


import Foundation
import BeanCountModel
import BeanCountParserUtils

enum EventParser {

    private static let regex: NSRegularExpression = {
        // swiftlint:disable:next force_try
        try! NSRegularExpression(pattern: "^\(DateParser.dateGroup)\\s+event\\s+\"([^\"]*)\"\\s+\"([^\"]*)\"\\s*(;.*)?$", options: [])
    }()

    static func parseFrom(line: String, metaData: [String: String] = [:]) -> Event? {
        let matches = line.matchingStrings(regex: self.regex)
        guard let match = matches[safe: 0], let date = DateParser.parseFrom(string: match[1]) else {
            return nil
        }
        return Event(date: date, name: match[2], value: match[3], metaData: metaData)
    }

}
