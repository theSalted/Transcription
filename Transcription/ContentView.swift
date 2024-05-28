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
                transcription
                if !whisper.bufferEnergy.isEmpty {
                    visualizer
                }
                recordButton
                
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
    
    var transcription: some View {
        ScrollView {
            VStack(alignment: .leading) {
                let confirmedText = whisper.confirmedText.first == " " ?
                    String(whisper.confirmedText.dropFirst()) :
                    whisper.confirmedText
                let hypothesisText = whisper.hypothesisText
                Text("\(Text(confirmedText))\(Text(hypothesisText).fontWeight(.bold).foregroundColor(.gray))")
                    .font(.headline)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity)
        .defaultScrollAnchor(.bottom)
        .textSelection(.enabled)
        .contentMargins(15, for: .scrollContent)
    }
    
    var visualizer: some View {
        ScrollView(.horizontal) {
            HStack(spacing: 1) {
                let startIndex = max(whisper.bufferEnergy.count - 300, 0)
                ForEach(Array(whisper.bufferEnergy.enumerated())[startIndex...], id: \.element) { _, energy in
                    ZStack {
                        RoundedRectangle(cornerRadius: 2)
                            .frame(width: 2, height: CGFloat(energy) * 24)
                    }
                    .frame(maxHeight: 24)
                    .background(energy > Float(whisper.settings.silenceThreshold) ? Color.green.opacity(0.2) : Color.red.opacity(0.2))
                }
            }
        }
        .defaultScrollAnchor(.trailing)
        .frame(height: 24)
        .scrollIndicators(.never)
    }
    
    var recordButton: some View {
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
}

#Preview {
    ContentView()
}

