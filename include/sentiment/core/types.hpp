#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sentiment {

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using sz = std::size_t;
using str = std::string;
using sv = std::string_view;

template <typename T>
using vec = std::vector<T>;

template <typename T, sz N>
using arr = std::array<T, N>;

enum class Sentiment : u8 {
    Negative = 0,
    Neutral  = 1,
    Positive = 2
};

struct Review {
    str text;
    Sentiment sentiment;
};

struct SparseVector {
    vec<u32> indices;
    vec<double> values;

    [[nodiscard]] bool empty() const noexcept {
        return indices.empty();
    }

    [[nodiscard]] sz size() const noexcept {
        return indices.size();
    }

    void clear() noexcept {
        indices.clear();
        values.clear();
    }

    void reserve(sz n) {
        indices.reserve(n);
        values.reserve(n);
    }
};

} // namespace sentiment
