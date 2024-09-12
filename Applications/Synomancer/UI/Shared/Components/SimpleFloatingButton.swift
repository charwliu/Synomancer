//
//  SimpleFloatingButton.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



import SwiftUI

struct SimpleFloatingButton: View {
    var systemImage: String
    var onClick: () -> ()
    
    var body: some View {
        Button(action: onClick) {
            Image(systemName: systemImage)
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .foregroundColor(Color.label)
                .frame(height: 18)
        }
        .buttonStyle(GrowingButton())
        .contentShape(Rectangle())
    }
}

#Preview {
    SimpleFloatingButton(systemImage: "photo.fill", onClick: {})
        .frame(width: 100, height: 100)
}
