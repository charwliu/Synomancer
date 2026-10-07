//
//  ReadingAloudView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//


import SwiftUI

struct ReadingAloudView: View {
    var isGenerating = false
    var generationStatus = "Generating Local Speech…"
    var onStopTap: () -> ()
    @State private var animationsRunning = false
    
    var body: some View {
        HStack {
            
            Image(systemName: "speaker.wave.3")
                .symbolEffect(.variableColor.iterative,  options: .repeat(100), value: animationsRunning)
                .scaledToFit()
                .frame(width: 18)
            
            Text(isGenerating ? generationStatus : "Reading Aloud")
                .font(.system(size: 14))
            
            Spacer()
            
            Button(action: onStopTap) {
                Image(systemName: "stop.fill")
                    .font(.system(size: 15, weight: .semibold))
                    .padding(5)
            }
            .buttonStyle(GrowingButton())
            .help("Stop Reading Aloud")
            .accessibilityLabel("Stop Reading Aloud")
            
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: 24).fill(.regularMaterial)
        }
        .padding()
        .onAppear {
            animationsRunning = true
        }
    }
}

#Preview {
    ReadingAloudView(onStopTap: {})
}
