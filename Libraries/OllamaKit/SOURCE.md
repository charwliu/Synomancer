# OllamaKit adaptation

Upstream: https://github.com/kevinhermawan/OllamaKit
Version: 5.0.8
Revision: b6ff6a0c75a3756450cfb38074f6264be1314abc
License: MIT; original LICENSE and source notices retained.

The asynchronous HTTP stream constrains its decoded response to `Decodable & Sendable`, matching the response types already marked Sendable upstream. This fixes Swift 6 transfer checking at AsyncThrowingStream. No endpoint or response format is changed.

Trailing source whitespace is normalized for the repository whitespace checks.
