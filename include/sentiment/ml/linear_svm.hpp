#include "sentiment/common/types.hpp"
#include "sentiment/core/types.hpp"

#pragma once

#include "sentiment/core/types.hpp"
#include <bits/stdc++.h>

namespace sentiment {

class LinearSVM {
public:
    explicit LinearSVM(
        sz feature_count
    );

    void set_weights(
        sz class_id,
        vec<double> weights
    );

    void set_bias(
        sz class_id,
        double bias
    );

    [[nodiscard]]
    Prediction predict(
        const vec<double>& features
    ) const noexcept;

private:
    static constexpr sz ClassCount = 3;

    sz feature_count_;

    vec<vec<double>> weights_;

    arr<double, ClassCount> bias_{};
};

} // namespace sentiment