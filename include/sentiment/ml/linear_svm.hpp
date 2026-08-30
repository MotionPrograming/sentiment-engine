#pragma once

#include "sentiment/core/types.hpp"

namespace sentiment {

class LinearSVM {
public:
    explicit LinearSVM(
        sz feature_count
    );

    void train(
        const vec<vec<double>>& features,
        const vec<Sentiment>& labels,
        sz epochs = 5,
        double learning_rate = 0.01,
        double regularization = 0.0001
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

    [[nodiscard]]
    bool trained() const noexcept;

private:
    static constexpr sz ClassCount = 3;

    sz feature_count_;

    vec<vec<double>> weights_;

    arr<double, ClassCount> bias_{};

    bool trained_{false};
};

} // namespace sentiment
