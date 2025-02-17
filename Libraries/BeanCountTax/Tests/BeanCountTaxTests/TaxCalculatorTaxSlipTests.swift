//
//  TaxCalculatorTaxSlipTests.swift
//  BeanCountTax
//
//  Created by Liu Wei on 9/30/24.
//

@testable import BeanCountTax
import Testing
import BeanCountModel

final class TaxCalculatorTaxSlipTests {
    @Test func generateTaxSlips() async throws {
        let ledger = try basicLedger()
        
        // Calculate tax slips
        let taxSlips = try TaxCalculator.generateTaxSlips(from: ledger, for: 2_024)
        
        // Check the generated tax slips
        #expect(taxSlips.count == 2)
        // Check tax slip 1
        let taxSlip1 = taxSlips[0]
        #expect(taxSlip1.name == "Taxslip1")
        #expect(taxSlip1.issuer == "Issuer 1")
        #expect(taxSlip1.year == 2_024)
        #expect(taxSlip1.boxes.count == 2)
        #expect(taxSlip1.rows.count == 1)
        
        // Check row in tax slip 1
        let row1 = taxSlip1.rows[0]
        #expect(row1.symbol == nil)
        #expect(row1.name == nil)
        
        // Check tax box 1 in tax slip 1
        #expect(row1.values[1].box == "TaxBox2")
        #expect(row1.values[1].value == Amount(number: 50, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax slip 2
        let taxSlip2 = taxSlips[1]
        #expect(taxSlip2.name == "Taxslip2")
        #expect(taxSlip2.issuer == nil)
        #expect(taxSlip2.year == 2_024)
        #expect(taxSlip2.boxes.count == 1)
        #expect(taxSlip2.rows.count == 1)
        
        // Check row in tax slip 2
        let row2 = taxSlip2.rows[0]
        #expect(row2.symbol == nil)
        #expect(row2.name == nil)
        
        // Check tax box 1 in tax slip 2
        #expect(row2.values[0].box == "TaxBox1")
        #expect(row2.values[0].value == Amount(number: 150, commoditySymbol: "EUR").multiCurrencyAmount)
        
        // No tax slips
        #expect(throws: (any Error).self) {try TaxCalculator.generateTaxSlips(from: ledger, for: 2022)}
        
        // No tax slip currency
        #expect(throws: (any Error).self) {try TaxCalculator.generateTaxSlips(from: ledger, for: 2023)}
    }
    
    @Test func generateTaxSlipsWithSymbol() throws {
        let taxSlips = try TaxCalculator.generateTaxSlips(from: try symbolLedger(), for: 2_024)
        #expect(taxSlips.count == 2)
        
        // Check tax slip 1
        let taxSlip1 = taxSlips[0]
        #expect(taxSlip1.name == "Taxslip1")
        #expect(taxSlip1.boxes.count == 2)
        #expect(taxSlip1.rows.count == 2)

        // Check row 1 in tax slip 1
        let row1 = taxSlip1.rows[0]
        #expect(row1.symbol == "SYM")
        #expect(row1.name == nil)
        
        // Check tax box 1 in row 1 in tax slip 1
        #expect(row1.values[0].box == "TaxBox1")
        #expect(row1.values[0].value == Amount(number: 100, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax box 2 in row 1 in tax slip 1
        #expect(row1.values[1].box == "TaxBox2")
        #expect(row1.values[1].value == MultiCurrencyAmount())
        
        // Check row 2 in tax slip 1
        let row2 = taxSlip1.rows[1]
        #expect(row2.symbol == "SYMB")
        #expect(row2.name == "DescB")
        
        // Check tax box 1 in row 2 tax slip 1
        #expect(row2.values[1].box == "TaxBox2")
        #expect(row2.values[1].value == Amount(number: 50, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax slip 2
        let taxSlip2 = taxSlips[1]
        #expect(taxSlip2.name == "Taxslip2")
        #expect(taxSlip2.boxes.count == 1)
        #expect(taxSlip2.rows.count == 1)
        
        // Check row 1 in tax slip 2
        let row3 = taxSlip2.rows[0]
        #expect(row3.symbol == "SYMBO")
        #expect(row3.name == "Desc")
        
        // Check tax box 1 in tax slip 3
        #expect(row3.values[0].box == "TaxBox1")
        #expect(row3.values[0].value == Amount(number: 150, commoditySymbol: "EUR").multiCurrencyAmount)
    }
    
    @Test func generateTaxSlipWithDifferentCurrencies() throws {
        
        let ledger = try currencyLedger()
        
        // Calculate tax slips
        let taxSlips = try TaxCalculator.generateTaxSlips(from: ledger, for: 2_024)
        
        // Check the generated tax slip
        #expect(taxSlips.count == 1)
        
        // Check tax slip 1
        let taxSlip = taxSlips[0]
        #expect(taxSlip.name == "Taxslip1")
        #expect(taxSlip.year == 2_024)
        #expect(taxSlip.boxes.count == 1)
        #expect(taxSlip.rows.count == 1)
        
        // Check row in tax slip 1
        let row1 = taxSlip.rows[0]
        #expect(row1.symbol == nil)
        #expect(row1.name == nil)
       
        // Check tax box 1 in tax slip 1
        #expect(row1.values[0].box == "TaxBox1")
        #expect(
            row1.values[0].value ==
            (Amount(number: 637.50, commoditySymbol: "USD").multiCurrencyAmount +
             Amount(number: 50, commoditySymbol: "JPY").multiCurrencyAmount)
        )
        #expect(
            row1.values[0].originalValue ==
            (Amount(number: 100, commoditySymbol: "USD").multiCurrencyAmount +
             Amount(number: 50, commoditySymbol: "JPY").multiCurrencyAmount +
             Amount(number: 200, commoditySymbol: "EUR").multiCurrencyAmount +
             Amount(number: 150, commoditySymbol: "CAD").multiCurrencyAmount)
        )
    }
    
    @Test func generateTaxSlipsWithSplitAccounts() throws {
        let taxSlips = try TaxCalculator.generateTaxSlips(from: splitAccountLedger(), for: 2_024)
        
        // Check the generated tax slips
        #expect(taxSlips.count == 2)
        
        // Check tax slip 1
        let taxSlip1 = taxSlips[0]
        #expect(taxSlip1.name == "Taxslip1")
        #expect(taxSlip1.issuer == "Issuer 1")
        #expect(taxSlip1.year == 2_024)
        #expect(taxSlip1.boxes.count == 3)
        #expect(taxSlip1.rows.count == 1)
        
        // Check row in tax slip 1
        let row1 = taxSlip1.rows[0]
        #expect(row1.symbol == nil)
        #expect(row1.name == nil)
        
        // Check tax box 1 in tax slip 1
        #expect(row1.values[0].box  == "TaxBox1")
        #expect(row1.values[0].value  == Amount(number: 128, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax box 2 in tax slip 1
        #expect(row1.values[1].box  == "TaxBox2")
        #expect(row1.values[1].value  == Amount(number: 50, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax box 3 in tax slip 1
        #expect(row1.values[2].box  == "SplitBox3")
        #expect(row1.values[2].value  == Amount(number: -112, commoditySymbol: "USD").multiCurrencyAmount)
        
        // Check tax slip 2
        let taxSlip2 = taxSlips[1]
        #expect(taxSlip2.name  == "Taxslip2")
        #expect(taxSlip2.issuer  == nil )
        #expect(taxSlip2.year == 2_024)
        #expect(taxSlip2.boxes.count == 1)
        #expect(taxSlip2.rows.count == 1)
        
        // Check row in tax slip 2
        let row2 = taxSlip2.rows[0]
        #expect(row2.symbol == nil)
        #expect(row2.name == nil)
        
        // Check tax box 1 in tax slip 2
        #expect(row2.values[0].box == "TaxBox1")
        #expect(row2.values[0].value == Amount(number: 150, commoditySymbol: "EUR").multiCurrencyAmount)
    }
}

