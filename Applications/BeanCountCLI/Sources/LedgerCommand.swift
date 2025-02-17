//
//  LedgerCommand.swift
//  BeanCountCLI
//
//  Created by Liu Wei on 9/30/24.
//
import ArgumentParser
import Foundation
import BeanCountModel
import BeanCountParser

struct LedgerCommandOptions: ParsableArguments {
    @Argument(help: "The file to parse.")
    var file: String
}

protocol LedgerCommand: ParsableCommand {
    var ledgerOptions: LedgerCommandOptions { get }
}

extension LedgerCommand {
    
    func parseLedger() throws -> Ledger {
        let ledger: Ledger
        do {
            ledger = try Parser.parse(contentOf: URL(fileURLWithPath: ledgerOptions.file))
        } catch {
            print(error.localizedDescription)
            throw ExitCode.failure
        }
        return ledger
    }
    
}

