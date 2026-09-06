#pragma once

#include "sentiment/core/types.hpp"
#include "sentiment/text/tfidf_vectorizer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sentiment {

class LinearSVM {
public:
    static constexpr std::size_t ClassCount = 3;

    static constexpr double MinimumClassWeight = 0.50;
    static constexpr double MaximumClassWeight = 5.00;
    static constexpr double MaximumGradientScale = 0.0050;
    static constexpr double BiasLearningRateScale = 0.10;
    static constexpr double Margin = 1.0;

    LinearSVM() = default;

    explicit LinearSVM(
        std::size_t feature_count
    );

    void set_class_weights(
        const std::array<double, ClassCount>& weights
    );

    const std::array<double, ClassCount>&
    class_weights() const noexcept;

    void train_sample(
        const SparseVector& features,
        Sentiment label,
        double learning_rate = 0.00035,
        double regularization = 0.00001
    );

    Sentiment predict(
        const SparseVector& features
    ) const;

    std::array<double, ClassCount>
    decision_scores(
        const SparseVector& features
    ) const;

    double confidence(
        const SparseVector& features
    ) const;

    std::size_t feature_count() const noexcept;

    bool trained() const noexcept;

    std::uint64_t training_steps() const noexcept;

    std::uint64_t margin_violations() const noexcept;

    void save(
        const std::string& path
    ) const;

    void load(
        const std::string& path
    );

private:
    std::array<std::vector<double>, ClassCount> weights_;
    std::array<double, ClassCount> bias_{};
    std::array<double, ClassCount> class_weights_{1.0, 1.0, 1.0};

    std::size_t feature_count_ = 0;
    bool trained_ = false;
    std::uint64_t training_steps_ = 0;
    std::uint64_t margin_violations_ = 0;
};

} // namespace sentiment