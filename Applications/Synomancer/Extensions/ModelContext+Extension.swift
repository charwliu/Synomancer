//
//  ModelContext+Extension.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//

import Foundation
import SwiftData

extension ModelContext {
    func saveChanges() throws {
        if self.hasChanges {
            try self.save()
        }
    }
}
