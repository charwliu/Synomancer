//
//  TestUtils.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

import Foundation
import BeanCountModel

enum TestUtils: Sendable {

    static let date20170610: Date = {
        Date(timeIntervalSince1970: 1_497_078_000)
    }()

    static let date20170609: Date = {
        Date(timeIntervalSince1970: 1_496_991_600)
    }()

    static let date20170608: Date = {
        Date(timeIntervalSince1970: 1_496_905_200)
    }()

    static let cad: CommoditySymbol = "CAD"
    static let eur: CommoditySymbol = "EUR"
    static let usd: CommoditySymbol = "USD"

    static let cadCommodity: Commodity = {
        Commodity(symbol: cad)
    }()

    static let eurCommodity: Commodity = {
        Commodity(symbol: eur)
    }()

    static let usdCommodity: Commodity = {
        Commodity(symbol: usd)
    }()

    static let chequing: AccountName = {
        try! AccountName("Assets:Chequing") // swiftlint:disable:this force_try
    }()

    static let cash: AccountName = {
        try! AccountName("Assets:Cash") // swiftlint:disable:this force_try
    }()

    static let income: AccountName = {
        try! AccountName("Income:Test") // swiftlint:disable:this force_try
    }()

    static let amount: Amount = {
        Amount(number: Decimal(1), commoditySymbol: cad)
    }()

    static let amount2: Amount = {
        Amount(number: Decimal(1), commoditySymbol: eur)
    }()

}
