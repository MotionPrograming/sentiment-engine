#pragma once

#include "sentiment/core/types.hpp"
#include <bits/stdc++.h>

namespace sentiment {

class Tokenizer {
public:
    [[nodiscard]]
    vec<str> tokenize(sv text) const;

private:
    static bool is_ascii_alpha(char c) noexcept;
    static char to_lower(char c) noexcept;
};

} // namespace sentiment