//
//  RemovableImage.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//

import SwiftUI

struct RemovableImage: View {
    var image: Image
    var onClick: () -> ()
    var height: Double = 80
    
    var body: some View {
        Button(action: {onClick() }) {
            ZStack(alignment: .topTrailing) {
                image
                    .resizable()
                    .scaledToFit()
                    .frame(height: height)
                
                Image(systemName: "x.circle.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 20, height: 20)
                    .padding(5)
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    RemovableImage(image: Image(systemName: "star"), onClick: {})
}
