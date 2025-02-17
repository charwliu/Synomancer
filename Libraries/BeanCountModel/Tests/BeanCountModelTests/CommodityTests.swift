//
//  CommodityTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing
import Foundation

struct CommodityTests {
    @Test func testDescription() async throws {
        let commodity = Commodity(symbol: TestUtils.cad)
        #expect(String(describing: commodity) == "")
    }
    
    @Test func testDescriptionSpecialCharactersOpening() async throws {
        let symbol = "💵"
        let string = "2017-06-08 commodity \(symbol)"
        let commodity = Commodity(symbol: symbol, opening: TestUtils.date20170608)
        #expect(String(describing: commodity) == string)
    }
    
    @Test func testDescriptionMetaData() async throws {
        let symbol = "CAD"
        let string = "2017-06-08 commodity \(symbol)\n  A: \"B\""
        let commodity = Commodity(symbol: symbol, opening: TestUtils.date20170608, metaData: ["A": "B"])
        #expect(String(describing: commodity) == string)
    }
    
    @Test func testValidate() async throws {
        let commodity = Commodity(symbol: "EUR", opening: TestUtils.date20170608)
        guard case .valid = commodity.validate() else {
            Issue.record("\(commodity) is not valid")
            return
        }
    }
    
    @Test func testValidateWithoutDate() async throws {
        let commodity = Commodity(symbol: "EUR")
        if case .invalid(let error) = commodity.validate() {
            #expect(error == "Commodity EUR does not have an opening date")
        } else {
            Issue.record("\(commodity) is valid")
        }
    }
    
    @Test func testEqual() async throws {
        var eur = Commodity(symbol: "EUR")
        var eur2 = Commodity(symbol: "EUR")
        let cad = Commodity(symbol: "CAD")
        #expect(eur == eur2)
        #expect(eur != cad)
        
        // meta data
        eur2 = Commodity(symbol: "EUR", metaData: ["A": "B"])
        #expect(eur != eur2)
        
        eur = Commodity(symbol: "EUR", metaData: ["A": "B"])
        #expect(eur == eur2)
    }
    
    @Test func testGreater() async throws {
        let eur = TestUtils.eur
        let cad = TestUtils.cad
        
        #expect(eur > cad)
        #expect(!(eur < cad))
        
        #expect(!(eur > eur))
        #expect(!(cad < cad))
    }
}
