//
//  StorageController.swift
//  Synomancer
//
//  Created by Liu Wei on 10/3/24.
//


import Foundation

class StorageController {
    private let accountsFileURL = URL.documentsDirectory
        .appendingPathComponent("Accounts")
        .appendingPathExtension("json")
    
    func save(_ accounts: [Account]) {
        guard let data = try? JSONEncoder().encode(accounts) else { return }
        try? data.write(to: accountsFileURL)
    }
    
    func fetchAccounts() -> [Account] {
        guard let data = try? Data(contentsOf: accountsFileURL) else { return [] }
        let accounts = try? JSONDecoder().decode([Account].self, from: data)
        return accounts ?? []
    }
}
