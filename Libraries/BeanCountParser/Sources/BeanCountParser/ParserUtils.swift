//
//  ParserUtils.swift
//  BeanCountParser
//
//  Created by Liu Wei on 9/27/24.
//


/// Constants for parsing a ledger file
enum ParserUtils {

    static let accountGroup = "([^\\s]+:[^\\s]+)"
    static let decimalGroup = "([-+]?[0-9]+(,[0-9]{3})*(.[0-9]+)?)"
    static let commodityGroup = "([^\\s]+)"
    static let amountGroup = "\(decimalGroup)\\s+\(commodityGroup)"

}
