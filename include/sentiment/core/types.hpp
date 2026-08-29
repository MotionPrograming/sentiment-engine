#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sentiment {

using sz  = std::size_t;
using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using str = std::string;
using sv  = std::string_view;

template <typename T>
using vec = std::vector<T>;

template <typename T, sz N>
using arr = std::array<T, N>;

enum class Sentiment : u8 {
    Negative = 0,
    Neutral  = 1,
    Positive = 2
};

[[nodiscard]]
inline constexpr sv to_string(Sentiment sentiment) noexcept {
    switch (sentiment) {
        case Sentiment::Negative:
            return "negative";

        case Sentiment::Neutral:
            return "neutral";

        case Sentiment::Positive:
            return "positive";
    }

    return "unknown";
}

struct Review {
    str text;
    Sentiment sentiment{Sentiment::Neutral};
};

struct Prediction {
    Sentiment label{Sentiment::Neutral};
    double score{0.0};
    arr<double, 3> probabilities{0.0, 0.0, 0.0};
};

struct BatchPredictionRequest {
    vec<str> texts;
};

struct BatchPredictionResult {
    vec<Prediction> predictions;
    u64 latency_us{0};
};

} // namespace sentiment
