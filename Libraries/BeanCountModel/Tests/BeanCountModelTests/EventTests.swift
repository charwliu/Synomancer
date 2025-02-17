//
//  EventTests.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//

@testable import BeanCountModel
import Testing

struct EventTests {
    @Test func testEqual() async throws {
        let event1 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        let event2 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        #expect(event1 == event2)
        #expect(!(event1 < event2))
        #expect(!(event2 < event1))
    }
    
    @Test func testEqualRespectsDate() async throws {
        let event1 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        let event2 = Event(date: TestUtils.date20170609, name: "A", value: "B")
        #expect(event1 != event2)
        #expect(event1 < event2)
        #expect(!(event2 < event1))
    }
    
    @Test func testEqualRespectsName() async throws {
        let event1 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        let event2 = Event(date: TestUtils.date20170608, name: "C", value: "B")
        #expect(event1 != event2)
        #expect(event1 < event2)
        #expect(!(event2 < event1))
    }
    
    @Test func testEqualRespectsValue() async throws {
        let event1 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        let event2 = Event(date: TestUtils.date20170608, name: "A", value: "C")
        #expect(event1 != event2)
        #expect(event1 < event2)
        #expect(!(event2 < event1))
    }
    
    @Test func testEqualRespectsMetaData() async throws{
        let event1 = Event(date: TestUtils.date20170608, name: "A", value: "B", metaData: ["A": "B"])
        var event2 = Event(date: TestUtils.date20170608, name: "A", value: "B")
        #expect(event1 != event2)
        #expect(!(event1 < event2))
        #expect(event2 < event1)
        event2 = Event(date: TestUtils.date20170608, name: "A", value: "B", metaData: ["A": "B"])
        #expect(event1 == event2)
        #expect(!(event1 < event2))
        #expect(!(event2 < event1))
    }
    
    @Test func testDescription() async throws {
        var event = Event(date: TestUtils.date20170608, name: "name", value: "B")
        #expect(String(describing: event) == "2017-06-08 event \"name\" \"B\"")
        event = Event(date: TestUtils.date20170608, name: "name", value: "B", metaData: ["A": "B"])
        #expect(String(describing: event) == "2017-06-08 event \"name\" \"B\"\n  A: \"B\"")
        
    }
    
}
