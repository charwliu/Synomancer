//
//  Date+Transaction.swift
//  Synomancer
//
//  Created by Liu Wei on 10/3/24.
//

import Foundation

extension Date {
    var transactionFormatted: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter.string(from: self)
    }
}

