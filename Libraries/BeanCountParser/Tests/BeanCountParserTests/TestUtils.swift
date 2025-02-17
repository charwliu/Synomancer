//
//  TestUtils.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


import Foundation
import BeanCountModel

enum TestUtils {

    static let calendar = Calendar.current

    static let date20170608: Date = {
        var dateComponents = DateComponents()
        dateComponents.year = 2_017
        dateComponents.month = 6
        dateComponents.day = 8
        dateComponents.timeZone = TimeZone.current
        return calendar.date(from: dateComponents)!
    }()

    static let date20170609: Date = {
        var dateComponents = DateComponents()
        dateComponents.year = 2_017
        dateComponents.month = 6
        dateComponents.day = 9
        dateComponents.timeZone = TimeZone.current
        return calendar.date(from: dateComponents)!
    }()

    static let date20170610: Date = {
        var dateComponents = DateComponents()
        dateComponents.year = 2_017
        dateComponents.month = 6
        dateComponents.day = 10
        dateComponents.timeZone = TimeZone.current
        return calendar.date(from: dateComponents)!
    }()

    static let cash: AccountName = {
        try! AccountName("Assets:Cash") // swiftlint:disable:this force_try
    }()
    
    static let checking: AccountName = {
        try! AccountName("Assets:Checking") // swiftlint:disable:this force_try
    }()
    
    static let special: AccountName = {
        try! AccountName("Assets:💰")
    }()
}
