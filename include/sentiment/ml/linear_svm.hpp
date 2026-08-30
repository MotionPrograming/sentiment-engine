#pragma once

#include "sentiment/core/types.hpp"

namespace sentiment {

class LinearSVM {
public:
    explicit LinearSVM(
        sz feature_count
    );

    void train_sample(
        const SparseVector& features,
        Sentiment label,
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
    Prediction predict(
        const SparseVector& features
    ) const noexcept;

    [[nodiscard]]
    const vec<vec<double>>& weights()
        const noexcept;

    [[nodiscard]]
    const arr<double, 3>& bias()
        const noexcept;

    [[nodiscard]]
    bool trained() const noexcept;

private:
    static constexpr sz ClassCount = 3;

    sz feature_count_;

    vec<vec<double>> weights_;

    arr<double, ClassCount> bias_{};

    /*
     * Lazy L2 regularization.
     *
     * last_update[class][feature] stores the training step
     * at which that coordinate was last physically shrunk.
     */
    vec<vec<u64>> last_update_;

    u64 step_{0};

    bool trained_{false};

    void apply_lazy_regularization(
        sz class_id,
        sz feature,
        double learning_rate,
        double regularization
    ) noexcept;

    [[nodiscard]]
    double effective_weight(
        sz class_id,
        sz feature,
        double learning_rate,
        double regularization
    ) const noexcept;
};

} // namespace sentiment
