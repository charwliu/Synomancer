//
//  SelectedImageView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//

import SwiftUI

struct SelectedImageView: View {
    @Binding var image: Image?
    
    private func onClick() {
        withAnimation(.snappy(duration: 0.3)) {
            image = nil
        }
    }
    
    var body: some View {
        if let selectedImage = image {
            Button(action: onClick) {
                selectedImage
                    .renderingMode(.original)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 40, maxHeight: 40)
                    .padding(.vertical, 4)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
    }
}

#Preview {
    SelectedImageView(image: .constant(Image(systemName: "photo")))
}
