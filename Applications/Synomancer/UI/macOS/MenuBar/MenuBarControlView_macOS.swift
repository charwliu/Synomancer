//
//  MenuBarControlView.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



#if os(macOS)
import SwiftUI

struct MenuBarControlView: View {
    var notifications: [NotificationMessage]
    var hasLocalModels: Bool
    var body: some View {
        VStack {
            Grid(horizontalSpacing: 12, verticalSpacing: 12) {
                GridRow {
                    HStack {
                        ControlView(
                            icon: hasLocalModels ? "checkmark.circle" : "x.circle",
                            title: "Chat models",
                            subtitle: hasLocalModels ? "Ready" : "Select Model"
                        )
                        ControlView(icon: "checkmark.circle", title: String(localized: "Synomancer"), subtitle: "Online")
                    }
                    .padding(12)
                    .background {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(.ultraThinMaterial)
                    }
                }
            }
            
            VStack(alignment: .leading) {
                Text("Events")
                    .font(.title3)
                
                ForEach(notifications) { notification in
                    HStack {
                        Text(notification.message)
                            .lineLimit(1)
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(notification.status == .info ? Color.blue.opacity(0.2) : Color.red.opacity(0.2)))
                    .listRowSeparator(.hidden)
                }
                .listStyle(PlainListStyle())
                .scrollContentBackground(.hidden)
            }
            .showIf(notifications.count > 0)
            .padding(.top, 10)
        }
        .padding()
        .frame(width: 360)
    }
    
    @ViewBuilder
    func ControlView(icon: String, title: String, subtitle: String? = nil) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.largeTitle)
                .symbolRenderingMode(.multicolor)
                .foregroundStyle(.gray)
            
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.callout)
                    .foregroundStyle(.primary)
                
                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(Color.grayCustom)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

extension MenuBarControlView {
    static let icon: Image = {
        guard let image = NSImage(named: "logo-nobg")?.copy() as? NSImage,
              image.size.width > 0, image.size.height > 0 else {
            return Image(systemName: "sparkles")
        }
        let aspectRatio = image.size.width / image.size.height
        image.size = NSSize(width: 18 * aspectRatio, height: 18)
        image.isTemplate = false

        return Image(nsImage: image)
    }()
}

#Preview {
    MenuBarControlView(notifications: NotificationMessage.sample, hasLocalModels: true)
}

#endif
