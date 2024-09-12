//
//  SidebarButton.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/9.
//



import SwiftUI

struct SidebarButton: View {
    var title: String
    var image: String
    var onClick: () -> ()
    
    var body: some View {
        Button(action: onClick) {
            HStack {
                Image(systemName: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 16)
                
                Text(title)
                    .lineLimit(1)
                    .font(.system(size: 14))
                    .fontWeight(.regular)
                
                Spacer()
            }
            .padding(8)
            .foregroundColor(Color(.label))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    SidebarButton(title: "Settings", image: "gearshape.fill", onClick: {})
}
