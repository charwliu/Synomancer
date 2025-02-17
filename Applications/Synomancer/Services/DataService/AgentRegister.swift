//
//  AgentRegister.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//

public protocol AgentRegister: Sendable {
    associatedtype AgentType: DataAgent
    var id: String { get }
    @Sendable func agent() async -> AgentType
}

extension AgentRegister { public var id: String { "\(AgentType.self)" } }
