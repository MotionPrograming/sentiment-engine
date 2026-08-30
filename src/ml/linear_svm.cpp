#include "sentiment/ml/linear_svm.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace sentiment {

LinearSVM::LinearSVM(sz feature_count)
    : feature_count_(feature_count),
      weights_(
          ClassCount,
          vec<double>(
              feature_count,
              0.0
          )
      ) {
}

void LinearSVM::train(
    const vec<vec<double>>& features,
    const vec<Sentiment>& labels,
    sz epochs,
    double learning_rate,
    double regularization
) {
    trained_ = false;

    if (features.empty() ||
        features.size() != labels.size() ||
        feature_count_ == 0 ||
        epochs == 0 ||
        learning_rate <= 0.0 ||
        regularization < 0.0) {
        return;
    }

    for (const auto& row : features) {

        if (row.size() != feature_count_) {
            return;
        }
    }

    // Start from zero weights.
    for (auto& class_weights : weights_) {
        std::fill(
            class_weights.begin(),
            class_weights.end(),
            0.0
        );
    }

    bias_.fill(0.0);

    std::vector<sz> order(features.size());

    for (sz i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    std::mt19937 rng(42);

    for (sz epoch = 0; epoch < epochs; ++epoch) {

        std::shuffle(
            order.begin(),
            order.end(),
            rng
        );

        for (const sz sample_index : order) {

            const auto& x =
                features[sample_index];

            const sz true_class =
                static_cast<sz>(labels[sample_index]);

            for (sz class_id = 0;
                 class_id < ClassCount;
                 ++class_id) {

                const double target =
                    class_id == true_class
                        ? 1.0
                        : -1.0;

                double score =
                    bias_[class_id];

                for (sz j = 0;
                     j < feature_count_;
                     ++j) {

                    score +=
                        weights_[class_id][j] *
                        x[j];
                }

                const double margin =
                    target * score;

                // L2 regularization.
                for (sz j = 0;
                     j < feature_count_;
                     ++j) {

                    weights_[class_id][j] *=
                        (1.0 -
                         learning_rate *
                         regularization);
                }

                if (margin < 1.0) {

                    for (sz j = 0;
                         j < feature_count_;
                         ++j) {

                        weights_[class_id][j] +=
                            learning_rate *
                            target *
                            x[j];
                    }

                    bias_[class_id] +=
                        learning_rate * target;
                }
            }
        }
    }

    trained_ = true;
}

void LinearSVM::set_weights(
    sz class_id,
    vec<double> weights
) {
    if (class_id >= ClassCount ||
        weights.size() != feature_count_) {
        return;
    }

    weights_[class_id] = std::move(weights);
    trained_ = true;
}

void LinearSVM::set_bias(
    sz class_id,
    double bias
) {
    if (class_id >= ClassCount) {
        return;
    }

    bias_[class_id] = bias;
}

Prediction LinearSVM::predict(
    const vec<double>& features
) const noexcept {

    Prediction prediction;

    if (features.size() != feature_count_) {
        return prediction;
    }

    arr<double, ClassCount> scores{};

    for (sz c = 0;
         c < ClassCount;
         ++c) {

        double score =
            bias_[c];

        const auto& weights =
            weights_[c];

        for (sz i = 0;
             i < feature_count_;
             ++i) {

            score +=
                features[i] *
                weights[i];
        }

        scores[c] = score;
    }

    const auto best =
        static_cast<sz>(
            std::distance(
                scores.begin(),
                std::max_element(
                    scores.begin(),
                    scores.end()
                )
            )
        );

    prediction.label =
        static_cast<Sentiment>(best);

    prediction.score =
        scores[best];

    return prediction;
}

bool LinearSVM::trained() const noexcept {
    return trained_;
}

} // namespace sentiment
