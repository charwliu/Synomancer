//
//  ValidationResult.swift
//  BeanCountModel
//
//  Created by Liu Wei on 9/26/24.
//


import Foundation

/// Result of a validation
///
/// - valid: the tested object is valid
/// - invalid: the tested object is invalid, including an error message
enum ValidationResult: Equatable {

    case valid
    case invalid(String)

}
