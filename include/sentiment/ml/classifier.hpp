#pragma once

#include "sentiment/core/types.hpp"
#include <bits/stdc++.h>

namespace sentiment {

// Forward declaration
struct Prediction;

class Classifier {
public:
    virtual ~Classifier() = default;

    [[nodiscard]]
    virtual Prediction predict(
        sv text
    ) const = 0;
};

} // namespace sentiment