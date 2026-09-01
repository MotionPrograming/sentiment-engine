#pragma once

#include "sentiment/core/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sentiment {

/*
 * ============================================================
 * Production Linear SVM
 * ============================================================
 *
 * Multiclass strategy:
 *
 *     Crammer-Singer style multiclass hinge loss
 *
 * Classes:
 *
 *     0 = Negative
 *     1 = Neutral
 *     2 = Positive
 *
 * Class weighting:
 *
 *     Negative = 1.0
 *     Neutral  = 4.5
 *     Positive = 1.0
 *
 * IMPORTANT:
 *
 * train_sample() intentionally has EXACTLY FOUR arguments.
 *
 *     train_sample(
 *         features,
 *         label,
 *         learning_rate,
 *         regularization
 *     );
 *
 * Class weights are configured separately using:
 *
 *     set_class_weights(...)
 *
 * ============================================================
 */

class LinearSVM {
public:

    static constexpr std::size_t ClassCount = 3;

    /*
     * Valid class-weight range.
     *
     * Neutral = 4.5 must not be clamped to 2.0.
     */
    static constexpr double MinimumClassWeight = 0.50;
    static constexpr double MaximumClassWeight = 5.00;

    /*
     * Gradient safety limit.
     *
     * Previous value 0.0025 could suppress the intended
     * effect of a 4.5x Neutral class weight.
     */
    static constexpr double MaximumGradientScale = 0.0050;

    /*
     * Bias updates are intentionally smaller than feature
     * updates for stability.
     */
    static constexpr double BiasLearningRateScale = 0.10;

    /*
     * Standard hinge-loss margin.
     */
    static constexpr double Margin = 1.0;

    LinearSVM() = default;

    explicit LinearSVM(
        std::size_t feature_count
    );

    /*
     * --------------------------------------------------------
     * Class weights
     * --------------------------------------------------------
     */

    void set_class_weights(
        const std::array<double, ClassCount>& weights
    );

    const std::array<double, ClassCount>&
    class_weights() const noexcept;

    /*
     * --------------------------------------------------------
     * Training
     * --------------------------------------------------------
     *
     * EXACTLY FOUR ARGUMENTS.
     */
    void train_sample(
        const SparseVector& features,
        Sentiment label,
        double learning_rate = 0.00035,
        double regularization = 0.00001
    );

    /*
     * --------------------------------------------------------
     * Prediction
     * --------------------------------------------------------
     */

    Sentiment predict(
        const SparseVector& features
    ) const;

    /*
     * Returns class decision scores.
     */
    std::array<double, ClassCount>
    decision_scores(
        const SparseVector& features
    ) const;

    /*
     * Confidence based on softmax-normalized decision scores.
     */
    double confidence(
        const SparseVector& features
    ) const;

    /*
     * --------------------------------------------------------
     * Model information
     * --------------------------------------------------------
     */

    std::size_t feature_count() const noexcept;

    bool trained() const noexcept;

    std::uint64_t training_steps() const noexcept;

    std::uint64_t margin_violations() const noexcept;

    /*
     * --------------------------------------------------------
     * Persistence
     * --------------------------------------------------------
     */

    void save(
        const std::string& path
    ) const;

    void load(
        const std::string& path
    );

private:

    /*
     * weights_[class][feature]
     */
    std::array<std::vector<double>, ClassCount>
        weights_;

    /*
     * One bias per class.
     */
    std::array<double, ClassCount>
        bias_{};

    /*
     * Per-class training weights.
     *
     * Default:
     *
     *     [1.0, 1.0, 1.0]
     */
    std::array<double, ClassCount>
        class_weights_{
            1.0,
            1.0,
            1.0
        };

    std::size_t feature_count_ = 0;

    bool trained_ = false;

    std::uint64_t training_steps_ = 0;

    std::uint64_t margin_violations_ = 0;
};

} // namespace sentiment