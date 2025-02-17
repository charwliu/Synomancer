//
//  Int+Cents.swift
//  Synomancer
//
//  Created by Liu Wei on 10/3/24.
//

import Foundation

extension Int {
    var centsFormatted: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .currency
        return formatter.string(from: NSNumber(value: Float(self) / 100 )) ?? ""
    }
}
