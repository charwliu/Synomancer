//
//  Test.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//
@testable import BeanCountModel
import Foundation
import Testing

struct AccountTests {
    
    @Test func testBookingMethod() async throws {
        let defaultAccount = Account(name:TestUtils.cash)
        #expect(defaultAccount.bookingMethod == .strict)
        
        let fifoAccount = Account(name: TestUtils.cash, bookingMethod: .fifo)
        #expect(fifoAccount.bookingMethod == .fifo)
        
        let lifoAccount = Account(name: TestUtils.cash, bookingMethod: .lifo)
        #expect(lifoAccount.bookingMethod == .lifo)
    }
    
    @Test func testDescription() async throws {
        var account = Account(name: TestUtils.cash)
        #expect(String(describing: account) == "")
        account = Account(name: TestUtils.cash, opening: TestUtils.date20170608)
        #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash)")
        account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: TestUtils.date20170608)
        #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \(TestUtils.eur)")
        account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: TestUtils.date20170608, closing: TestUtils.date20170609)
        #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \(TestUtils.eur)\n2017-06-09 close \(TestUtils.cash)")
        account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: TestUtils.date20170608, closing: TestUtils.date20170609, metaData: ["A": "B"])
        #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \(TestUtils.eur)\n  A: \"B\"\n2017-06-09 close \(TestUtils.cash)")
    }
    
    @Test func testDescriptionBookingMethod() async throws {
        for bookingMethod in [BookingMethod.fifo, BookingMethod.lifo] {
            var account = Account(name: TestUtils.cash, bookingMethod: bookingMethod)
            #expect(String(describing: account) == "")
            account = Account(name: TestUtils.cash, bookingMethod: bookingMethod, opening: TestUtils.date20170608)
            #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \"\(bookingMethod)\"")
            account = Account(name: TestUtils.cash, bookingMethod: bookingMethod, commoditySymbol: TestUtils.eur, opening: TestUtils.date20170608)
            #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \(TestUtils.eur) \"\(bookingMethod)\"")
            account = Account(name: TestUtils.cash,
                              bookingMethod: bookingMethod,
                              commoditySymbol: TestUtils.eur,
                              opening: TestUtils.date20170608,
                              closing: TestUtils.date20170609)
            #expect(String(describing: account) == "2017-06-08 open \(TestUtils.cash) \(TestUtils.eur) \"\(bookingMethod)\"\n2017-06-09 close \(TestUtils.cash)")
        }
    }
    
    @Test func testDescriptionSpecialCharacters() async throws {
        let accountNameSpecial = try AccountName("Assets:💰")
        var account = Account(name: accountNameSpecial)
        #expect(String(describing: account) == "")
        account = Account(name: accountNameSpecial, opening: TestUtils.date20170608)
        #expect(String(describing: account) == "2017-06-08 open \(accountNameSpecial)")
        let symbol = "💵"
        account = Account(name: accountNameSpecial, commoditySymbol: symbol, opening: TestUtils.date20170608)
        #expect(String(describing: account) == "2017-06-08 open \(accountNameSpecial) \(symbol)")
        account = Account(name: accountNameSpecial, commoditySymbol: symbol, opening: TestUtils.date20170608, closing: TestUtils.date20170609)
        #expect(String(describing: account) == "2017-06-08 open \(accountNameSpecial) \(symbol)\n2017-06-09 close \(accountNameSpecial)")
    }
    
    @Test func testIsPostingValid_NotOpenPast() async throws {
        let account = Account(name: TestUtils.cash)
        let posting = Posting(accountName: TestUtils.cash, amount: Amount(number: Decimal(1), commoditySymbol: TestUtils.eur))
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .invalid("""
            2017-06-08 * "Payee" "Narration"
              Assets:Cash 1 EUR was posted while the accout Assets:Cash was closed
            """)
                
        )
    }
    
    @Test func testIsPostingValid_NoOpenPresent() async throws {
        let account = Account(name: TestUtils.cash)
        let posting = Posting(accountName: TestUtils.cash, amount: Amount(number: Decimal(1), commoditySymbol: TestUtils.eur))
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .invalid("""
            2017-06-08 * "Payee" "Narration"
              Assets:Cash 1 EUR was posted while the accout Assets:Cash was closed
            """)
        )
    }
    
    @Test func testIsPostingValid_BeforeOpening() async throws {
        let account = Account(name: TestUtils.cash, opening: TestUtils.date20170609)
        let posting = Posting(accountName: TestUtils.cash, amount: Amount(number: Decimal(1), commoditySymbol: TestUtils.eur))
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .invalid("""
            2017-06-08 * "Payee" "Narration"
              Assets:Cash 1 EUR was posted while the accout Assets:Cash was closed
            """)
        )
    }
    
    @Test func testIsPostingValid_AfterOpening() async throws {
        let account = Account(name: TestUtils.cash, opening: TestUtils.date20170609)
        let posting1 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction1 = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                       postings: [posting1])
        #expect(account.validate(transaction1.postings[0]) == .valid)
        
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction2 = Transaction(metaData: TransactionMetaData(date: Date(), payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                       postings: [posting2])
        #expect(account.validate(transaction2.postings[0]) == .valid)
    }
    
    
    @Test func testIsPostingValid_BeforeClosing() async throws{
        let account = Account(name: TestUtils.cash, opening: TestUtils.date20170609, closing: TestUtils.date20170609)
        let posting = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) ==  .valid)
    }
    
    @Test func testIsPostingValid_AfterClosing() async throws {
        let account = Account(name: TestUtils.cash, opening: TestUtils.date20170609, closing: TestUtils.date20170609)
        let posting = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170610, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .invalid("""
            2017-06-10 * "Payee" "Narration"
              Assets:Cash 1 CAD was posted while the accout Assets:Cash was closed
            """)
        )
    }
    
    @Test func testIsPostingValid_WithoutCommodity() async throws {
        let account = Account(name: TestUtils.cash, opening: TestUtils.date20170608)
        let posting1 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction1 = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                       postings: [posting1])
        #expect(account.validate(transaction1.postings[0]) == .valid)
        
        let posting2 = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction2 = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                       postings: [posting2])
        #expect(account.validate(transaction2.postings[0]) == .valid)
    }
    
    @Test func testIsPostingValid_CorrectCommodity() async throws {
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.amount.commoditySymbol, opening: TestUtils.date20170608)
        let posting = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .valid)
    }
    
    @Test func nametestIsPostingValid_WrongCommodity() async throws {
        let account = Account(name: TestUtils.cash, commoditySymbol: "\(TestUtils.amount.commoditySymbol)1", opening: TestUtils.date20170608)
        let posting = Posting(accountName: TestUtils.cash, amount: TestUtils.amount)
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "Payee", narration: "Narration", flag: Flag.complete, tags: []),
                                      postings: [posting])
        #expect(account.validate(transaction.postings[0]) == .invalid("""
            2017-06-09 * "Payee" "Narration"
              Assets:Cash 1 CAD uses a wrong commodiy for account Assets:Cash - Only CAD1 is allowed
            """)
        )
    }
    
    @Test func testIsValid() async throws {
        var account = Account(name: TestUtils.cash)
        
        // neither closing nor opening
        #expect(account.validate() == .valid)
        
        // only opening
        account = Account(name: TestUtils.cash, opening: TestUtils.date20170608)
        #expect(account.validate() == .valid)
        
        // Closing == opening
        account = Account(name: TestUtils.cash, opening: TestUtils.date20170608, closing: TestUtils.date20170608)
        #expect(account.validate() == .valid)
        
        // Closing > opening
        account = Account(name: TestUtils.cash, opening: TestUtils.date20170608, closing: TestUtils.date20170609)
        #expect(account.validate() == .valid)
        
        // Closing < opening
        account = Account(name: TestUtils.cash, opening: TestUtils.date20170609, closing: TestUtils.date20170608)
        #expect(account.validate() == .invalid("Account Assets:Cash was closed on 2017-06-08 before it was opened on 2017-06-09"))
        
        // only closing
        account = Account(name: TestUtils.cash, closing: TestUtils.date20170608)
        #expect(account.validate() == .invalid("Account Assets:Cash has a closing date but no opening"))
    }
    
    @Test func testValidateBalance() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        try ledger.add(account)
        
        account.balances.append(Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: Amount(number: 0, commoditySymbol: TestUtils.cad)))
        #expect(account.validateBalance(in: ledger) == .valid)
        
        account.balances.append(Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.cad)))
        #expect(account.validateBalance(in: ledger) == .invalid("Balance failed for 2017-06-09 balance Assets:Cash 1 CAD - 1 CAD too much (0 tolerance)"))
        
        var posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.cad))
        var transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        #expect(account.validateBalance(in: ledger) == .valid)
        
        posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 10, commoditySymbol: TestUtils.cad))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        account.balances.append(Balance(date: TestUtils.date20170610, accountName: TestUtils.cash, amount: Amount(number: 11, commoditySymbol: TestUtils.cad)))
        #expect(account.validateBalance(in: ledger) == .valid)
    }
    
    @Test func testValidateBalanceEmpty() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        try ledger.add(account)
        
        #expect(account.validateBalance(in: ledger) == .valid)
        
        let posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.cad))
        let transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        #expect(account.validateBalance(in: ledger) == .valid)
    }
    
    @Test func testValidateBalanceDifferentCommodity() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash)
        try ledger.add(account)
        
        account.balances.append(Balance(date: TestUtils.date20170608, accountName: TestUtils.cash, amount: Amount(number: 0, commoditySymbol: TestUtils.eur)))
        
        account.balances.append(Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.cad)))
        #expect(account.validateBalance(in: ledger) == .invalid("Balance failed for 2017-06-09 balance Assets:Cash 1 CAD - 1 CAD too much (0 tolerance)"))
        
        var posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.cad))
        var transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        #expect(account.validateBalance(in: ledger) == .valid)
        
        // Ignores other commodity without currency
        posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.usd))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        #expect(account.validateBalance(in: ledger) == .valid)
        
        account.balances.append(Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.eur)))
        posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1, commoditySymbol: TestUtils.eur))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        #expect(account.validateBalance(in: ledger) == .valid)
    }
    
    @Test func testValidateBalanceTolerance() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        try ledger.add(account)
        
        var posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 1.1, commoditySymbol: TestUtils.cad, decimalDigits: 1))
        var transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        account.balances = [Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1.15, commoditySymbol: TestUtils.cad, decimalDigits: 2))]
        #expect(account.validateBalance(in: ledger) == .invalid("Balance failed for 2017-06-09 balance Assets:Cash 1.15 CAD - 0.05 CAD too much (0.005 tolerance)"))
        
        account.balances = [Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1.15, commoditySymbol: TestUtils.cad, decimalDigits: 1))]
        #expect(account.validateBalance(in: ledger) == .valid)
        
        posting = Posting(accountName: TestUtils.cash, amount: Amount(number: 0.055, commoditySymbol: TestUtils.cad, decimalDigits: 3))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        account.balances = [Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1.16, commoditySymbol: TestUtils.cad, decimalDigits: 2))]
        #expect(account.validateBalance(in: ledger) == .invalid("Balance failed for 2017-06-09 balance Assets:Cash 1.16 CAD - 0.005 CAD too much (0.0005 tolerance)"))
        
        account.balances = [
            Balance(date: TestUtils.date20170609, accountName: TestUtils.cash, amount: Amount(number: 1.155_5, commoditySymbol: TestUtils.cad, decimalDigits: 3))
        ]
        #expect(account.validateBalance(in: ledger) == .valid)
    }
    
    @Test func testValidateInventoryEmpty() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        try ledger.add(account)
        
        #expect(account.validateInventory(in: ledger) == .valid)
    }
    
    @Test func testValidateInventory() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        try ledger.add(account)
        
        var posting = Posting(accountName: TestUtils.cash,
                              amount: Amount(number: 1.1, commoditySymbol: TestUtils.cad, decimalDigits: 1),
                              price: nil,
                              cost: try Cost(amount: Amount(number: 5, commoditySymbol: TestUtils.cad), date: nil, label: "1"))
        var transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        posting = Posting(accountName: TestUtils.cash,
                          amount: Amount(number: 1.1, commoditySymbol: TestUtils.cad, decimalDigits: 1),
                          price: nil,
                          cost: try Cost(amount: Amount(number: 5, commoditySymbol: TestUtils.cad), date: nil, label: nil))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        posting = Posting(accountName: TestUtils.cash,
                          amount: Amount(number: -1, commoditySymbol: TestUtils.cad, decimalDigits: 0),
                          price: nil,
                          cost: try Cost(amount: Amount(number: 5, commoditySymbol: TestUtils.cad), date: nil, label: "1"))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        #expect(account.validateInventory(in: ledger) ==  .valid)
    }
    
    @Test func testValidateInvalidInventory() async throws {
        let ledger = Ledger()
        let account = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        let amount = Amount(number: 1.1, commoditySymbol: TestUtils.cad, decimalDigits: 1)
        let cost = try Cost(amount: Amount(number: 5, commoditySymbol: TestUtils.cad), date: nil, label: "1")
        try ledger.add(account)
        
        var posting = Posting(accountName: TestUtils.cash, amount: amount, price: nil, cost: cost)
        var transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170608, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        posting = Posting(accountName: TestUtils.cash, amount: amount, price: nil, cost: cost)
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170609, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        posting = Posting(accountName: TestUtils.cash,
                          amount: Amount(number: -1.0, commoditySymbol: TestUtils.cad, decimalDigits: 0),
                          price: nil,
                          cost: try Cost(amount: Amount(number: 5, commoditySymbol: TestUtils.cad), date: nil, label: nil))
        transaction = Transaction(metaData: TransactionMetaData(date: TestUtils.date20170610, payee: "", narration: "", flag: .complete, tags: []), postings: [posting])
        ledger.add(transaction)
        
        #expect(account.validateInventory(in: ledger) == .invalid("""
            Ambigious Booking: -1 CAD {5 CAD}, matches: 1.1 CAD {2017-06-08, 5 CAD, "1"}
            1.1 CAD {2017-06-09, 5 CAD, "1"}, inventory: 1.1 CAD {2017-06-08, 5 CAD, "1"}
            1.1 CAD {2017-06-09, 5 CAD, "1"}
            """)
        )
    }
    
    @Test func testEqualName() async throws {
        let account1 = Account(name: TestUtils.cash)
        let account2 = Account(name: TestUtils.chequing)
        #expect(account1 != account2)
    }
   
    @Test func testEqualProperties() async throws{
        let date1 = TestUtils.date20170608
        let date2 = TestUtils.date20170609
        
        var account1 = Account(name: TestUtils.cash)
        var account2 = Account(name: TestUtils.cash)
        
        // equal
        #expect(account1 == account2)
        
        account1 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, closing: date1)
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, closing: date1)
        
        // equal
        #expect(account1 == account2)
        // different meta data
        account1 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, metaData: ["A": "B"])
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, metaData: ["A": "C"])
        #expect(account1 != account2)
        // same meta data
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, metaData: ["A": "B"])
        #expect(account1 == account2)
        // different commodity
        account1 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur)
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.cad)
        #expect(account1 != account2)
        // different opening
        account1 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1)
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date2)
        #expect(account1 != account2)
        // different closing
        account2 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date2, closing: date2)
        #expect(account1 != account2)
        account1 = Account(name: TestUtils.cash, commoditySymbol: TestUtils.eur, opening: date1, closing: date1)
        #expect(account1 != account2)
    }
     
}
