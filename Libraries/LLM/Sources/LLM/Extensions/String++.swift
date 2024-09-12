//
//  String++.swift
//  MLXLLM
//
//  Created by Liu Wei on 9/1/24.
//

import Swift

extension String {
    func deletingPrefix(_ prefix: String) -> String {
        guard hasPrefix(prefix) else {
            return self
        }
        return String(dropFirst(prefix.count))
    }
}


