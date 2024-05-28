//
//  WhisperView.swift
//  NotesAIMockUp
//
//  Created by Yuhao Chen on 5/28/24.
//

//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import SwiftUI
import WhisperKit
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import AVFoundation
import CoreML

/// A processor for handling Whisper-related tasks, including model loading, recording, and transcription.
final class WhisperProcessor {
    static let shared = WhisperProcessor()
    
    private(set) var whisperKit: WhisperKit?
    private(set) var modelState: ModelState = .unloaded
    private(set) var localModels: [String] = []
    private(set) var localModelPath: String = ""
    private(set) var availableModels: [String] = []
    private(set) var availableLanguages: [String] = []
    private(set) var loadingProgressValue: Float = 0.0
    private(set) var specializationProgressRatio: Float = 0.7
    private(set) var isRecording: Bool = false
    private(set) var isTranscribing: Bool = false
    private(set) var confirmedSegments: [TranscriptionSegment] = []
    private(set) var unconfirmedSegments: [TranscriptionSegment] = []
    private(set) var currentText: String = ""
    private(set) var currentChunks: [Int: (chunkText: [String], fallbacks: Int)] = [:]
    
    // UI-related states
    var bufferEnergy: [Float] = []
    var bufferSeconds: Double = 0
    var totalInferenceTime: TimeInterval = 0
    var tokensPerSecond: TimeInterval = 0
    var effectiveRealTimeFactor: TimeInterval = 0
    var effectiveSpeedFactor: TimeInterval = 0
    var currentEncodingLoops: Int = 0
    var currentLag: TimeInterval = 0
    var lastConfirmedSegmentEndSeconds: Float = 0
    var requiredSegmentsForConfirmation: Int = 4
    var prevWords: [WordTiming] = []
    var lastAgreedWords: [WordTiming] = []
    var confirmedWords: [WordTiming] = []
    var confirmedText: String = ""
    var hypothesisWords: [WordTiming] = []
    var hypothesisText: String = ""
    
    @AppStorage("selectedModel") private var selectedModel: String = WhisperKit.recommendedModels().default
    @AppStorage("repoName") private var repoName: String = "argmaxinc/whisperkit-coreml"
    @AppStorage("selectedLanguage") private var selectedLanguage: String = "english"
    @AppStorage("enableTimestamps") private var enableTimestamps: Bool = true
    @AppStorage("enablePromptPrefill") private var enablePromptPrefill: Bool = true
    @AppStorage("enableCachePrefill") private var enableCachePrefill: Bool = true
    @AppStorage("enableSpecialCharacters") private var enableSpecialCharacters: Bool = false
    @AppStorage("enableEagerDecoding") private var enableEagerDecoding: Bool = false
    @AppStorage("temperatureStart") private var temperatureStart: Double = 0
    @AppStorage("fallbackCount") private var fallbackCount: Double = 5
    @AppStorage("compressionCheckWindow") private var compressionCheckWindow: Double = 60
    @AppStorage("sampleLength") private var sampleLength: Double = 224
    @AppStorage("silenceThreshold") private var silenceThreshold: Double = 0.3
    @AppStorage("useVAD") private var useVAD: Bool = true
    @AppStorage("tokenConfirmationsNeeded") private var tokenConfirmationsNeeded: Double = 2
    @AppStorage("chunkingStrategy") private var chunkingStrategy: ChunkingStrategy = .none
    @AppStorage("encoderComputeUnits") private var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    @AppStorage("decoderComputeUnits") private var decoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    
    // Internal state variables
    private var firstTokenTime: TimeInterval = 0
    private var pipelineStart: TimeInterval = 0
    private var currentFallbacks: Int = 0
    private var currentDecodingLoops: Int = 0
    private var lastBufferSize: Int = 0
    private var transcriptionTask: Task<Void, Never>? = nil
    private var eagerResults: [TranscriptionResult?] = []
    private var prevResult: TranscriptionResult?
    private var lastAgreedSeconds: Float = 0.0
    
    private init() {
        fetchModels()
    }
    
    /// Fetch available models from the local storage and remote repository.
    func fetchModels() {
        availableModels = [selectedModel]
        
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let modelPath = documents.appendingPathComponent("huggingface/models/argmaxinc/whisperkit-coreml").path
            
            if FileManager.default.fileExists(atPath: modelPath) {
                localModelPath = modelPath
                do {
                    let downloadedModels = try FileManager.default.contentsOfDirectory(atPath: modelPath)
                    for model in downloadedModels where !localModels.contains(model) {
                        localModels.append(model)
                    }
                } catch {
                    print("Error enumerating files at \(modelPath): \(error.localizedDescription)")
                }
            }
        }
        
        localModels = WhisperKit.formatModelFiles(localModels)
        for model in localModels {
            if !availableModels.contains(model) {
                availableModels.append(model)
            }
        }
        
        Task {
            let remoteModels = try await WhisperKit.fetchAvailableModels(from: repoName)
            for model in remoteModels {
                if !availableModels.contains(model) {
                    availableModels.append(model)
                }
            }
        }
    }
    
    /// Load the specified model. Optionally redownloads the model if necessary.
    func loadModel(_ model: String, redownload: Bool = false) {
        whisperKit = nil
        Task {
            whisperKit = try await WhisperKit(
                computeOptions: getComputeOptions(),
                verbose: true,
                logLevel: .debug,
                prewarm: false,
                load: false,
                download: false
            )
            guard let whisperKit = whisperKit else { return }
            
            var folder: URL?
            
            if localModels.contains(model) && !redownload {
                folder = URL(fileURLWithPath: localModelPath).appendingPathComponent(model)
            } else {
                folder = try await WhisperKit.download(variant: model, from: repoName, progressCallback: { progress in
                    DispatchQueue.main.async {
                        self.loadingProgressValue = Float(progress.fractionCompleted) * self.specializationProgressRatio
                        self.modelState = .downloading
                    }
                })
            }
            
            await MainActor.run {
                self.loadingProgressValue = self.specializationProgressRatio
                self.modelState = .downloaded
            }
            
            if let modelFolder = folder {
                whisperKit.modelFolder = modelFolder
                
                await MainActor.run {
                    self.loadingProgressValue = self.specializationProgressRatio
                    self.modelState = .prewarming
                }
                
                let progressBarTask = Task {
                    await self.updateProgressBar(targetProgress: 0.9, maxTime: 240)
                }
                
                do {
                    try await whisperKit.prewarmModels()
                    progressBarTask.cancel()
                } catch {
                    progressBarTask.cancel()
                    if !redownload {
                        loadModel(model, redownload: true)
                        return
                    } else {
                        modelState = .unloaded
                        return
                    }
                }
                
                await MainActor.run {
                    self.loadingProgressValue = self.specializationProgressRatio + 0.9 * (1 - self.specializationProgressRatio)
                    self.modelState = .loading
                }
                
                try await whisperKit.loadModels()
                
                await MainActor.run {
                    if !self.localModels.contains(model) {
                        self.localModels.append(model)
                    }
                    
                    self.availableLanguages = Constants.languages.map { $0.key }.sorted()
                    self.loadingProgressValue = 1.0
                    self.modelState = whisperKit.modelState
                }
            }
        }
    }
    
    /// Get compute options based on user settings.
    private func getComputeOptions() -> ModelComputeOptions {
        return ModelComputeOptions(audioEncoderCompute: encoderComputeUnits, textDecoderCompute: decoderComputeUnits)
    }
    
    /// Update the progress bar with a target progress and maximum time.
    private func updateProgressBar(targetProgress: Float, maxTime: TimeInterval) async {
        let initialProgress = loadingProgressValue
        let decayConstant = -log(1 - targetProgress) / Float(maxTime)
        
        let startTime = Date()
        
        while true {
            let elapsedTime = Date().timeIntervalSince(startTime)
            
            let decayFactor = exp(-decayConstant * Float(elapsedTime))
            let progressIncrement = (1 - initialProgress) * (1 - decayFactor)
            let currentProgress = initialProgress + progressIncrement
            
            await MainActor.run {
                loadingProgressValue = currentProgress
            }
            
            if currentProgress >= targetProgress {
                break
            }
            
            do {
                try await Task.sleep(nanoseconds: 100_000_000)
            } catch {
                break
            }
        }
    }
    
    /// Transcribe audio samples in eager mode.
    func transcribeEagerMode(_ samples: [Float]) async throws -> TranscriptionResult? {
        guard let whisperKit = whisperKit else { return nil }
        
        guard whisperKit.textDecoder.supportsWordTimestamps else {
            confirmedText = "Eager mode requires word timestamps, which are not supported by the current model: \(selectedModel)."
            return nil
        }
        
        let languageCode = Constants.languages[selectedLanguage, default: Constants.defaultLanguageCode]
        let task: DecodingTask = .transcribe
        
        let options = DecodingOptions(
            verbose: true,
            task: task,
            language: languageCode,
            temperature: Float(temperatureStart),
            temperatureFallbackCount: Int(fallbackCount),
            sampleLength: Int(sampleLength),
            usePrefillPrompt: enablePromptPrefill,
            usePrefillCache: enableCachePrefill,
            skipSpecialTokens: !enableSpecialCharacters,
            withoutTimestamps: !enableTimestamps,
            wordTimestamps: true,
            firstTokenLogProbThreshold: -1.5
        )
        
        let decodingCallback: ((TranscriptionProgress) -> Bool?) = { progress in
            DispatchQueue.main.async {
                let fallbacks = Int(progress.timings.totalDecodingFallbacks)
                if progress.text.count < self.currentText.count {
                    if fallbacks == self.currentFallbacks {
                    } else {
                        print("Fallback occured: \(fallbacks)")
                    }
                }
                self.currentText = progress.text
                self.currentFallbacks = fallbacks
                self.currentDecodingLoops += 1
            }
            let currentTokens = progress.tokens
            let checkWindow = Int(self.compressionCheckWindow)
            if currentTokens.count > checkWindow {
                let checkTokens: [Int] = currentTokens.suffix(checkWindow)
                let compressionRatio = compressionRatio(of: checkTokens)
                if compressionRatio > options.compressionRatioThreshold! {
                    return false
                }
            }
            if progress.avgLogprob! < options.logProbThreshold! {
                return false
            }
            
            return nil
        }
        
        Logging.info("[EagerMode] \(lastAgreedSeconds)-\(Double(samples.count) / 16000.0) seconds")
        
        let streamingAudio = samples
        var streamOptions = options
        streamOptions.clipTimestamps = [lastAgreedSeconds]
        let lastAgreedTokens = lastAgreedWords.flatMap { $0.tokens }
        streamOptions.prefixTokens = lastAgreedTokens
        do {
            let transcription: TranscriptionResult? = try await whisperKit.transcribe(audioArray: streamingAudio, decodeOptions: streamOptions, callback: decodingCallback).first
            await MainActor.run {
                var skipAppend = false
                if let result = transcription {
                    self.hypothesisWords = result.allWords.filter { $0.start >= self.lastAgreedSeconds }
                    
                    if let prevResult = self.prevResult {
                        self.prevWords = prevResult.allWords.filter { $0.start >= self.lastAgreedSeconds }
                        let commonPrefix = findLongestCommonPrefix(self.prevWords, self.hypothesisWords)
                        Logging.info("[EagerMode] Prev \"\((self.prevWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Next \"\((self.hypothesisWords.map { $0.word }).joined())\"")
                        Logging.info("[EagerMode] Found common prefix \"\((commonPrefix.map { $0.word }).joined())\"")
                        
                        if commonPrefix.count >= Int(self.tokenConfirmationsNeeded) {
                            self.lastAgreedWords = commonPrefix.suffix(Int(self.tokenConfirmationsNeeded))
                            self.lastAgreedSeconds = self.lastAgreedWords.first!.start
                            Logging.info("[EagerMode] Found new last agreed word \"\(self.lastAgreedWords.first!.word)\" at \(self.lastAgreedSeconds) seconds")
                            
                            self.confirmedWords.append(contentsOf: commonPrefix.prefix(commonPrefix.count - Int(self.tokenConfirmationsNeeded)))
                            let currentWords = self.confirmedWords.map { $0.word }.joined()
                            Logging.info("[EagerMode] Current:  \(self.lastAgreedSeconds) -> \(Double(samples.count) / 16000.0) \(currentWords)")
                        } else {
                            Logging.info("[EagerMode] Using same last agreed time \(self.lastAgreedSeconds)")
                            skipAppend = true
                        }
                    }
                    self.prevResult = result
                }
                
                if !skipAppend {
                    self.eagerResults.append(transcription)
                }
            }
        } catch {
            Logging.error("[EagerMode] Error: \(error)")
        }
        
        await MainActor.run {
            let finalWords = self.confirmedWords.map { $0.word }.joined()
            self.confirmedText = finalWords
            
            let lastHypothesis = self.lastAgreedWords + findLongestDifferentSuffix(self.prevWords, self.hypothesisWords)
            self.hypothesisText = lastHypothesis.map { $0.word }.joined()
        }
        
        let mergedResult = mergeTranscriptionResults(eagerResults, confirmedWords: confirmedWords)
        
        return mergedResult
    }
    
    /// Toggle recording state.
    func toggleRecording(shouldLoop: Bool) {
        isRecording.toggle()
        
        if isRecording {
            resetState()
            startRecording(shouldLoop)
        } else {
            stopRecording(shouldLoop)
        }
    }
    
    /// Reset the state of the processor.
    private func resetState() {
        transcriptionTask?.cancel()
        isRecording = false
        isTranscribing = false
        whisperKit?.audioProcessor.stopRecording()
        currentText = ""
        currentChunks = [:]
        
        pipelineStart = Double.greatestFiniteMagnitude
        firstTokenTime = Double.greatestFiniteMagnitude
        effectiveRealTimeFactor = 0
        effectiveSpeedFactor = 0
        totalInferenceTime = 0
        tokensPerSecond = 0
        currentLag = 0
        currentFallbacks = 0
        currentEncodingLoops = 0
        currentDecodingLoops = 0
        lastBufferSize = 0
        lastConfirmedSegmentEndSeconds = 0
        requiredSegmentsForConfirmation = 2
        bufferEnergy = []
        bufferSeconds = 0
        confirmedSegments = []
        unconfirmedSegments = []
        
        eagerResults = []
        prevResult = nil
        lastAgreedSeconds = 0.0
        prevWords = []
        lastAgreedWords = []
        confirmedWords = []
        confirmedText = ""
        hypothesisWords = []
        hypothesisText = ""
    }
    
    /// Start recording audio.
    private func startRecording(_ loop: Bool) {
        if let audioProcessor = whisperKit?.audioProcessor {
            Task(priority: .userInitiated) {
                guard await AudioProcessor.requestRecordPermission() else {
                    print("Microphone access was not granted.")
                    return
                }
                
                try? audioProcessor.startRecordingLive { _ in
                    DispatchQueue.main.async {
                        self.bufferEnergy = self.whisperKit?.audioProcessor.relativeEnergy ?? []
                        self.bufferSeconds = Double(self.whisperKit?.audioProcessor.audioSamples.count ?? 0) / Double(WhisperKit.sampleRate)
                    }
                }
                
                isRecording = true
                isTranscribing = true
                if loop {
                    realtimeLoop()
                }
            }
        }
    }
    
    /// Stop recording audio.
    private func stopRecording(_ loop: Bool) {
        isRecording = false
        stopRealtimeTranscription()
        whisperKit?.audioProcessor.stopRecording()
        
        if !loop {
            Task {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                }
            }
        }
    }
    
    /// Real-time transcription loop.
    private func realtimeLoop() {
        transcriptionTask = Task {
            while isRecording && isTranscribing {
                do {
                    try await transcribeCurrentBuffer()
                } catch {
                    print("Error: \(error.localizedDescription)")
                    break
                }
            }
        }
    }
    
    /// Stop real-time transcription.
    private func stopRealtimeTranscription() {
        isTranscribing = false
        transcriptionTask?.cancel()
    }
    
    /// Transcribe the current audio buffer.
    private func transcribeCurrentBuffer() async throws {
        guard let whisperKit = whisperKit else { return }
        
        let currentBuffer = whisperKit.audioProcessor.audioSamples
        
        let nextBufferSize = currentBuffer.count - lastBufferSize
        let nextBufferSeconds = Float(nextBufferSize) / Float(WhisperKit.sampleRate)
        
        guard nextBufferSeconds > 1 else {
            await MainActor.run {
                if currentText.isEmpty {
                    currentText = "Waiting for speech..."
                }
            }
            try await Task.sleep(nanoseconds: 100_000_000)
            return
        }
        
        if useVAD {
            let voiceDetected = AudioProcessor.isVoiceDetected(
                in: whisperKit.audioProcessor.relativeEnergy,
                nextBufferInSeconds: nextBufferSeconds,
                silenceThreshold: Float(silenceThreshold)
            )
            guard voiceDetected else {
                await MainActor.run {
                    if currentText.isEmpty {
                        currentText = "Waiting for speech..."
                    }
                }
                
                try await Task.sleep(nanoseconds: 100_000_000)
                return
            }
        }
        
        lastBufferSize = currentBuffer.count
        
        let transcription = try await transcribeEagerMode(Array(currentBuffer))
        await MainActor.run {
            currentText = ""
            guard let segments = transcription?.segments else {
                return
            }
            
            self.tokensPerSecond = transcription?.timings.tokensPerSecond ?? 0
            self.firstTokenTime = transcription?.timings.firstTokenTime ?? 0
            self.pipelineStart = transcription?.timings.pipelineStart ?? 0
            self.currentLag = transcription?.timings.decodingLoop ?? 0
            self.currentEncodingLoops += Int(transcription?.timings.totalEncodingRuns ?? 0)
            let totalAudio = Double(currentBuffer.count) / Double(WhisperKit.sampleRate)
            self.totalInferenceTime += transcription?.timings.fullPipeline ?? 0
            self.effectiveRealTimeFactor = Double(self.totalInferenceTime) / totalAudio
            self.effectiveSpeedFactor = totalAudio / Double(self.totalInferenceTime)
            
            if segments.count > self.requiredSegmentsForConfirmation {
                let numberOfSegmentsToConfirm = segments.count - self.requiredSegmentsForConfirmation
                let confirmedSegmentsArray = Array(segments.prefix(numberOfSegmentsToConfirm))
                let remainingSegments = Array(segments.suffix(self.requiredSegmentsForConfirmation))
                
                if let lastConfirmedSegment = confirmedSegmentsArray.last, lastConfirmedSegment.end > self.lastConfirmedSegmentEndSeconds {
                    self.lastConfirmedSegmentEndSeconds = lastConfirmedSegment.end
                    if !self.confirmedSegments.contains(confirmedSegmentsArray) {
                        self.confirmedSegments.append(contentsOf: confirmedSegmentsArray)
                    }
                }
                self.unconfirmedSegments = remainingSegments
            } else {
                self.unconfirmedSegments = segments
            }
        }
    }
}
