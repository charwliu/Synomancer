//
//  LedgerManager.swift
//  Synomancer
//
//  Created by Liu Wei on 9/28/24.
//

import CryptoKit
import Foundation
import BeanCountModel
import BeanCountParser


enum LedgerManagerError: Error {
    case noLedgerSpecified
    case noTask
    case noAccess
}

@Observable
class LedgerManager: @unchecked Sendable {
    var url: URL? {
        didSet {
            if oldValue != url {
                Task { @MainActor in
                    do {
                        try await loadLedger()
                    } catch {
                        logger.error("\(error.localizedDescription)")
                    }
                }
            }
        }
    }
    
    var displayLedgerSelector = false
    var waitingForLedgerLoad = false
    private(set) var loadingLedger = false
    
    private var ledgerLoaded = false
    
    var ledger: Ledger? = nil
    var hash: SHA256.Digest? = nil
    
    // paramters only used for previews
    @MainActor
    init( _ ledgerURL: URL? = nil) {
        self.url = ledgerURL
    }
    
    /// Re-reads the ledger file from disk and checks if the hash
    /// changed. If so it will parse again, otherwise it won't.
    /// If the loading is stil ongoing, it cancles and re-starts.
    @MainActor
    public func refresh() {
        if self.hash != nil {
            Task { @MainActor in
                do {
                    let (newHash, _) = try self.getHashAndText()
                    if hash != newHash {
                        logger.info("Found file change during refresh")
                        try await self.loadLedger()
                    } else {
                        logger.info("No file change found during refresh")
                    }
                } catch {
                    logger.error("\(error.localizedDescription)")
                }
            }
        } else {
            Task { @MainActor in
                do {
                    try await loadLedger()
                } catch {
                    logger.error("\(error.localizedDescription)")
                }
            }
        }
    }
    
    /// Gets the content of the currently selected ledger.
    /// It  can check the hash of the current file to the loaded
    /// one, so it does not return content older then when the
    /// method was called.
    /// If parsing needs to happen, it will set the waitingForLedgerLoad variable.
    /// - Parameter skipHashCheck:Skip loading the current file to check the hash
    /// - Returns: Ledger
    public func getLedgerContent(skipHashCheck: Bool = false) async throws -> Ledger {
        guard url != nil else {
            throw LedgerManagerError.noLedgerSpecified
        }
        if !ledgerLoaded || skipHashCheck {
            DispatchQueue.main.async {
                self.waitingForLedgerLoad = true
            }
        }
        defer {
            DispatchQueue.main.async {
                self.waitingForLedgerLoad = false
            }
        }
        let (hash, ledger) = try await loadLedger()
        if skipHashCheck {
            return ledger
        }
        let (newHash, _) = try getHashAndText()
        if hash != newHash {
            logger.info("File changed")
            try await loadLedger()
            return try await getLedgerContent(skipHashCheck: true)
        }
        return ledger
    }
    
    @discardableResult
    private func loadLedger() async throws -> (SHA256.Digest, Ledger) {
        let (hash, text) = try self.getHashAndText()
        let ledger = Parser.parse(string: text)
        logger.debug("Done parsing")
        self.ledger = ledger
        self.hash = hash
        return (hash, ledger)
    }
    
    private func getHashAndText() throws -> (SHA256.Digest, String) {
        guard let url else {
            throw LedgerManagerError.noLedgerSpecified
        }
        let gotAccess = url.startAccessingSecurityScopedResource()
        if !gotAccess {
            throw LedgerManagerError.noAccess
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        url.stopAccessingSecurityScopedResource()
        let data = Data(text.utf8)
        let hash = SHA256.hash(data: data)
        return (hash, text)
    }
}


extension SHA256.Digest: @unchecked @retroactive Sendable {
    
}
