//
//  CostTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing
import Foundation

struct CostTests {
    private let label1 = "1"
    private let label2 = "2"
    
    @Test func testNegativeAmount() async throws {
        #expect { try Cost(amount: Amount(number: -1, commoditySymbol: TestUtils.eur), date: nil, label: nil)} throws: {
            $0.localizedDescription == "Invalid Cost, negative amount: {-1 EUR}"
        }
    }
    
    @Test func testEqual() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(cost1 == cost2)
    }
    
    @Test func testEqualRespectsAmount() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        let cost2 = try Cost(amount: TestUtils.amount2, date: TestUtils.date20170608, label: label1)
        let cost3 = try Cost(amount: nil, date: TestUtils.date20170608, label: label1)
        #expect(cost1 != cost2)
        #expect(cost1 != cost3)
    }
    
    @Test func testEqualRespectsDate() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label2)
        let cost3 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: nil)
        #expect(cost1 != cost2)
        #expect(cost1 != cost3)
    }
    
    @Test func testEqualWorksWithNil() async throws {
        let cost1 = try Cost(amount: nil, date: nil, label: nil)
        let cost2 = try Cost(amount: nil, date: nil, label: nil)
        #expect(cost1 == cost2)
    }
    
    @Test func testMatches() async throws {
        let cost1 = try Cost(amount: nil, date: nil, label: nil)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(cost1.matches(cost: cost2))
    }
    
    @Test func testMatchesLabel() async throws {
        let cost1 = try Cost(amount: nil, date: TestUtils.date20170608, label: nil)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(cost1.matches(cost: cost2))
    }
    
    @Test func testMatchesAmount() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: nil, label: nil)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(cost1.matches(cost: cost2))
    }
    
    @Test func testNotMatchesLabel() async throws {
        let cost1 = try Cost(amount: nil, date: nil, label: label2)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testNotMatchesDate() async throws {
        let cost1 = try Cost(amount: nil, date: TestUtils.date20170609, label: nil)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testNotMatchesAmount() async throws {
        let cost1 = try Cost(amount: TestUtils.amount2, date: nil, label: nil)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testMatchesLabelWrong() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label2)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testMatchesDateWrong() async throws {
        let cost1 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170609, label: label1)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testMatchesAmountWrong() async throws {
        let cost1 = try Cost(amount: TestUtils.amount2, date: TestUtils.date20170608, label: label1)
        let cost2 = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(!cost1.matches(cost: cost2))
    }
    
    @Test func testDescription() async throws {
        let cost = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: label1)
        #expect(String(describing: cost) == "{2017-06-08, \(String(describing: TestUtils.amount)), \"\(label1)\"}")
    }
    
    @Test func testDescriptionWithoutDate() async throws {
        let cost = try Cost(amount: TestUtils.amount, date: nil, label: label1)
        #expect(String(describing: cost) == "{\(String(describing: TestUtils.amount)), \"\(label1)\"}")
    }
    
    @Test func testDescriptionWithoutAmount() async throws {
        let cost = try Cost(amount: nil, date: TestUtils.date20170608, label: label1)
        #expect(String(describing: cost) == "{2017-06-08, \"\(label1)\"}")
    }
    
    @Test func testDescriptionWithoutLabel() async throws {
        let cost = try Cost(amount: TestUtils.amount, date: TestUtils.date20170608, label: nil)
        #expect(String(describing: cost) == "{2017-06-08, \(String(describing: TestUtils.amount))}")
    }
    
    @Test func testDescriptionWithOnlyAmount() async throws {
        let cost = try Cost(amount: TestUtils.amount, date: nil, label: nil)
        #expect(String(describing: cost) == "{\(String(describing: TestUtils.amount))}")
    }
    
    @Test func testDescriptionWithOnlyDate() async throws {
        let cost = try Cost(amount: nil, date: TestUtils.date20170608, label: nil)
        #expect(String(describing: cost) == "{2017-06-08}")
    }
    
    @Test func testDescriptionWithOnlyLabel() async throws {
        let cost = try Cost(amount: nil, date: nil, label: label1)
        #expect(String(describing: cost) == "{\"\(label1)\"}")
    }
    
    @Test func testEmptyDescription() async throws {
        let cost = try Cost(amount: nil, date: nil, label: nil)
        #expect(String(describing: cost) == "{}")
    }
    
}
