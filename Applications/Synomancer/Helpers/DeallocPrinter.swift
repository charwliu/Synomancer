//
//  DeallocPrinter.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//



import Foundation

class DeallocPrinter {
    var message: String
    
    init(_ message: String) {
        self.message = message
    }
    
    deinit {
        print("deallocated \(message)")
    }
}
