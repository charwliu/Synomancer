//
//  MenuBarControl.swift
//  Synomancer
//
//  Created by Liu Wei on 2024/9/10.
//



#if os(macOS)
import SwiftUI

struct MenuBarControl: View {
    @State private var appStore = AppStore.shared
    var body: some View {
        MenuBarControlView(notifications: appStore.notifications)
    }
}
#endif
