//
//  Check.swift
//  BeanCountCLI
//
//  Created by Liu Wei on 9/30/24.
//
import ArgumentParser
import Foundation
import Rainbow
import BeanCountModel


struct Check: LedgerCommand, ColorizedCommand {

    nonisolated(unsafe) static var configuration = CommandConfiguration(abstract: "Parses a ledger and prints any errors it finds")

    @OptionGroup()
    var ledgerOptions: LedgerCommandOptions
    @ArgumentParser.Flag(name: [.short, .long], help: "Don't print errors, only indicate via exit code.")
    private var quiet = false
    @OptionGroup()
    var colorOptions: ColorizedCommandOptions

    func run() throws {
//        adjustColorization()
        let ledger = try parseLedger()
        let errors = ledger.errors
        if !errors.isEmpty {
            if !quiet {
                print("Found ".red + String(errors.count).bold.red + " errors:\n".red)
                print(errors.joined(separator: "\n"))
            }
            throw ExitCode(65)
        } else if !quiet {
            print("No errors found.".green)
        }
    }

}
