#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <array>

namespace sentiment {

using sz  = std::size_t;
using sv  = std::string_view;
using str = std::string;

template<typename T>
using vec = std::vector<T>;

template<typename T, sz N>
using arr = std::array<T, N>;

}
