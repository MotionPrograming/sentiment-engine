#pragma once

#include <bits/stdc++.h>

#define u8  std::uint8_t
#define u64 std::uint64_t
#define sz  std::size_t

namespace sentiment::config {

// Model Configuration
inline constexpr sz kClassCount = 3;

inline constexpr u8 kNegativeLabel = 0;
inline constexpr u8 kNeutralLabel  = 1;
inline constexpr u8 kPositiveLabel = 2;

// TF-IDF Configuration
inline constexpr sz kMaxFeatures = 50'000;

// Ignore terms appearing in fewer than this many documents.
inline constexpr sz kMinDocumentFrequency = 3;

// Use unigram + bigram features.
inline constexpr sz kNgramMin = 1;
inline constexpr sz kNgramMax = 2;

// Text Processing Configuration
inline constexpr sz kInitialTokenCapacity = 16;
inline constexpr sz kInitialTokenSize = 32;

// Training Configuration
inline constexpr sz kMaxIterations = 2'000;
inline constexpr double kLearningRate = 0.01;
inline constexpr double kRegularization = 0.0001;

// Dataset Configuration
inline constexpr double kTestSplit = 0.20;
inline constexpr u64 kRandomSeed = 42;

// Performance Configuration
inline constexpr sz kDefaultBatchSize = 64;

} // namespace sentiment::config