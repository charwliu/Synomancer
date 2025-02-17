//
//  OptionTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing

struct OptionTests {
    @Test func testDescription() async throws {
        let option = Option(name: "name", value: "value1")
        #expect(String(describing: option) == "option \"name\" \"value1\"")
        
        let optionSpecialCharacters = Option(name: "😂", value: "😀")
        #expect(String(describing: optionSpecialCharacters) == "option \"😂\" \"😀\"")
    }
    
    @Test func testComparable() async throws {
        let option1 = Option(name: "name", value: "value1")
        let option2 = Option(name: "name", value: "value1")
        let option3 = Option(name: "name1", value: "value1") // check name
        let option4 = Option(name: "name", value: "value2") // check value
        
        #expect(option1 == option2)
        #expect(option1 != option3)
        #expect(option1 != option4)
    }
}
