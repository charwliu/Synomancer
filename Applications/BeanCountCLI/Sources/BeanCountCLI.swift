// The Swift Programming Language
// https://docs.swift.org/swift-book
// 
// Swift Argument Parser
// https://swiftpackageindex.com/apple/swift-argument-parser/documentation

import ArgumentParser

@main
struct BeanCountCLI: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "swiftbeancount",
        abstract: "A CLI tool for BeanCount",
        version: "0.1.0",
        subcommands: [Check.self, Stats.self, Accounts.self, TaxSlips.self]
    )
    public init() {
        
    }
    
}
