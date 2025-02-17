//
//  DataAgent.swift
//  Synomancer
//
//  Created by Liu Wei on 9/21/24.
//


#if canImport(SwiftData)
  public import Foundation
  public protocol DataAgent: Sendable {
    var agentID: UUID { get }
    func onUpdate(_ update: any DatabaseChangeSet)
    func onCompleted(_ closure: @Sendable @escaping () -> Void)
    func finish() async
  }

  extension DataAgent { public func onCompleted(_: @Sendable @escaping () -> Void) {} }
#endif
