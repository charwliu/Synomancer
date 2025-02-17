//
//  Previews.swift
//  Synomancer
//
//  Created by Liu Wei on 10/3/24.
//

import Foundation

extension Account {
    static let preview = Account(
        name: "Test account",
        iban: .generateIBAN(),
        kind: .checking
    )
}

extension Transaction {
    static let preview = Transaction(
        amount: 123456,
        beneficiary: "Salary",
        date: Date()
    )
}
