//
//  Transaction.swift
//  Synomancer
//
//  Created by Liu Wei on 10/3/24.
//


import Foundation

struct Transaction: Identifiable, Hashable {
	let id = UUID()
	let amount: Int
	let beneficiary: String
	let date: Date
}

extension Transaction: Codable {
	enum CodingKeys: CodingKey {
		case id
		case amount
		case beneficiary
		case date
	}
}