//
//  ContentView.swift
//  Transcription
//
//  Created by Yuhao Chen on 5/29/24.
//

import SwiftUI

struct ContentView: View {
    @State var whisper = WhisperProcessor.shared
    @State var showSettingsSheet = false
    
    var body: some View {
        NavigationStack {
            VStack {
                Spacer()
                Button {
                    withAnimation {
                        whisper.toggleRecording(shouldLoop: true)
                    }
                } label: {
                    ZStack {
                        Circle()
                            .stroke(.gray, lineWidth: 3)
                        if whisper.modelState != .loaded {
                            ProgressView()
                                .padding(5)
                        } else if !whisper.isRecording {
                            Circle()
                                .foregroundStyle(.red)
                                .padding(5)
                        } else {
                            RoundedRectangle(cornerRadius: 5)
                                .foregroundStyle(.red)
                                .frame(width: 25, height: 25)
                        }
                        
                    }
                }
                .disabled(whisper.modelState != .loaded)
                .frame(width: 60, height: 60)
            }
            .navigationTitle("Transcription")
            .toolbar {
                ToolbarItem() {
                    Button("Show More", systemImage: "ellipsis.circle") {
                        showSettingsSheet = true
                    }
                }
            }
        }
        .sheet(isPresented: $showSettingsSheet) {
            NavigationStack {
                List {
                    Section("Whisper Model") {
                        let model = whisper.settings.selectedModel
                        let state = whisper.modelState
                        HStack {
                            Text(model)
                            Spacer()
                            Text(state.description)
                                .foregroundStyle(.secondary)
                        }
                        if state != .loaded {
                            ProgressView(value: whisper.loadingProgressValue, total: 1.0)
                        }
                    }
                    
                }
                .navigationTitle("Settings")
            }
            .presentationDragIndicator(.visible)
        }
        
    }
}

#Preview {
    ContentView()
}

