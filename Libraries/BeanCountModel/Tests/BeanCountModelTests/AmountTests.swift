//
//  AmountTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//
@testable import BeanCountModel
import Testing
import Foundation

struct AmountTests {
    @Test func testEqual() async throws {
        #expect(TestUtils.amount == Amount(number: Decimal(1), commoditySymbol: TestUtils.cad))
    }
    
    @Test func testEqualRespectsAmount() async throws {
        #expect(TestUtils.amount != Amount(number: Decimal(10), commoditySymbol: TestUtils.cad))
    }
    
    @Test func testEqualRespectsCommodity() async throws {
        #expect(TestUtils.amount != Amount(number: Decimal(1.0), commoditySymbol: TestUtils.eur, decimalDigits: 1))
    }
    
    @Test func testDescriptionInteger() async throws {
        let amountInteger = 123
        let amount = Amount(number: Decimal(amountInteger), commoditySymbol: TestUtils.cad)
        
        #expect(String(describing: amount) == "\(amountInteger) \(TestUtils.cad)")
    }
    
    @Test func testDescriptionThousandsSeperator() async throws {
        let amountInteger = 1_234_567_890.00
        let amount = Amount(number: Decimal(amountInteger), commoditySymbol: TestUtils.cad, decimalDigits: 2)
        #expect(String(describing: amount) == "1,234,567,890.00 \(TestUtils.cad)")
    }
    
    @Test func testDescriptionLongFloat() async throws {
        let amount = Amount(number: Decimal(0.000_976_562_5), commoditySymbol: TestUtils.cad, decimalDigits: 10)
        #expect(String(describing: amount) == "0.0009765625 \(TestUtils.cad)")
    }
    
    @Test func testMultiCurrencyAmount() async throws {
        let decimal = Decimal(10)
        let amount = Amount(number: decimal, commoditySymbol: TestUtils.eur)
        #expect(amount.multiCurrencyAmount.amounts == [TestUtils.eur: decimal])
        #expect(amount.multiCurrencyAmount.decimalDigits == [TestUtils.eur: 0])
    }
    
    @Test func testMultiCurrencyAmountDecimalDigits() async throws {
        let decimal = Decimal(10.25)
        let amount = Amount(number: decimal, commoditySymbol: TestUtils.eur, decimalDigits: 2)
        #expect(amount.multiCurrencyAmount.amounts == [TestUtils.eur: decimal])
        #expect(amount.multiCurrencyAmount.decimalDigits == [TestUtils.eur: 2])
    }
}
