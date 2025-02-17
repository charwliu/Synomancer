//
//  TagTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing

struct TagTests {
    @Test func testDescription() async throws {
        let string = "String"
        let tag = Tag(name: string)
        #expect(String(describing: tag) == "#" + string)
    }
    
    @Test func testDescriptionSpecialCharacters() async throws {
        let string = "#️⃣"
        let tag = Tag(name: string)
        #expect(String(describing: tag) == "#" + string)
    }
    
    @Test func testEqual() async throws {
        let string1 = "String1"
        let string2 = "String2"
        let tag1 = Tag(name: string1)
        let tag2 = Tag(name: string1)
        let tag3 = Tag(name: string2)
        
        #expect(tag1 == tag2)
        #expect(tag1 != tag3)
        #expect(tag2 != tag3)
    }
    
    @Test func testGreater() async throws {
        let string1 = "A"
        let string2 = "B"
        let tag1 = Tag(name: string1)
        let tag2 = Tag(name: string2)
        
        #expect(tag1 < tag2)
        #expect(!(tag1 > tag2))
        #expect(!(tag1 > tag1))
        #expect(!(tag2 < tag2))
    }
}
