//
//  WhisperSettings.swift
//  Transcription
//
//  Created by Yuhao Chen on 5/29/24.
//

import WhisperKit
import SwiftUI
import CoreML

struct WhisperSettings {
    @AppStorage("selectedModel") var selectedModel: String = WhisperKit.recommendedModels().default
    @AppStorage("repoName") var repoName: String = "argmaxinc/whisperkit-coreml"
    @AppStorage("selectedLanguage") var selectedLanguage: String = "english"
    @AppStorage("enableTimestamps") var enableTimestamps: Bool = true
    @AppStorage("enablePromptPrefill") var enablePromptPrefill: Bool = true
    @AppStorage("enableCachePrefill") var enableCachePrefill: Bool = true
    @AppStorage("enableSpecialCharacters") var enableSpecialCharacters: Bool = false
    @AppStorage("enableEagerDecoding") var enableEagerDecoding: Bool = false
    @AppStorage("temperatureStart") var temperatureStart: Double = 0
    @AppStorage("fallbackCount") var fallbackCount: Double = 5
    @AppStorage("compressionCheckWindow")  var compressionCheckWindow: Double = 60
    @AppStorage("sampleLength") var sampleLength: Double = 224
    @AppStorage("silenceThreshold")  var silenceThreshold: Double = 0.3
    @AppStorage("useVAD") var useVAD: Bool = true
    @AppStorage("tokenConfirmationsNeeded")  var tokenConfirmationsNeeded: Double = 2
    @AppStorage("chunkingStrategy") var chunkingStrategy: ChunkingStrategy = .none
    @AppStorage("encoderComputeUnits") var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    @AppStorage("decoderComputeUnits") var decoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
}
