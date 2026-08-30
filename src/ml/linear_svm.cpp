#include "sentiment/ml/linear_svm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
      ),
      last_update_(
          ClassCount,
          vec<u64>(
              feature_count,
              0
          )
      ) {
}

void LinearSVM::apply_lazy_regularization(
    sz class_id,
    sz feature,
    double learning_rate,
    double regularization
) noexcept {

    if (class_id >= ClassCount ||
        feature >= feature_count_ ||
        regularization <= 0.0 ||
        learning_rate <= 0.0) {
        return;
    }

    const u64 last =
        last_update_[class_id][feature];

    if (last == step_) {
        return;
    }

    const u64 elapsed =
        step_ - last;

    const double shrink =
        std::pow(
            1.0 -
                learning_rate *
                regularization,
            static_cast<double>(elapsed)
        );

    weights_[class_id][feature] *= shrink;

    last_update_[class_id][feature] =
        step_;
}

double LinearSVM::effective_weight(
    sz class_id,
    sz feature,
    double learning_rate,
    double regularization
) const noexcept {

    if (class_id >= ClassCount ||
        feature >= feature_count_) {
        return 0.0;
    }

    if (regularization <= 0.0 ||
        learning_rate <= 0.0) {
        return weights_[class_id][feature];
    }

    const u64 last =
        last_update_[class_id][feature];

    if (last == step_) {
        return weights_[class_id][feature];
    }

    const u64 elapsed =
        step_ - last;

    const double shrink =
        std::pow(
            1.0 -
                learning_rate *
                regularization,
            static_cast<double>(elapsed)
        );

    return
        weights_[class_id][feature] *
        shrink;
}

void LinearSVM::train_sample(
    const SparseVector& features,
    Sentiment label,
    double learning_rate,
    double regularization
) {

    trained_ = false;

    if (!features.valid() ||
        features.indices.empty() ||
        feature_count_ == 0 ||
        learning_rate <= 0.0 ||
        regularization < 0.0) {
        return;
    }

    const sz true_class =
        static_cast<sz>(label);

    if (true_class >= ClassCount) {
        return;
    }

    ++step_;

    /*
     * For each one-vs-rest classifier.
     */
    for (sz class_id = 0;
         class_id < ClassCount;
         ++class_id) {

        const double target =
            class_id == true_class
                ? 1.0
                : -1.0;

        double score =
            bias_[class_id];

        /*
         * Sparse dot product.
         *
         * Only active TF-IDF coordinates are touched.
         */
        for (sz i = 0;
             i < features.indices.size();
             ++i) {

            const sz index =
                features.indices[i];

            if (index >= feature_count_) {
                continue;
            }

            score +=
                effective_weight(
                    class_id,
                    index,
                    learning_rate,
                    regularization
                ) *
                features.values[i];
        }

        const double margin =
            target * score;

        if (margin < 1.0) {

            for (sz i = 0;
                 i < features.indices.size();
                 ++i) {

                const sz index =
                    features.indices[i];

                if (index >= feature_count_) {
                    continue;
                }

                /*
                 * Bring this coordinate up to the
                 * current lazy regularization step.
                 */
                apply_lazy_regularization(
                    class_id,
                    index,
                    learning_rate,
                    regularization
                );

                /*
                 * Hinge-loss update.
                 */
                weights_[class_id][index] +=
                    learning_rate *
                    target *
                    features.values[i];
            }

            bias_[class_id] +=
                learning_rate *
                target;
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

    weights_[class_id] =
        std::move(weights);

    std::fill(
        last_update_[class_id].begin(),
        last_update_[class_id].end(),
        step_
    );

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

    if (!trained_ ||
        features.size() != feature_count_) {
        return prediction;
    }

    arr<double, ClassCount> scores{};

    /*
     * This prediction overload is mainly useful for
     * compatibility/tests. Sparse prediction should be
     * preferred for the real pipeline.
     */
    constexpr double default_learning_rate = 0.01;
    constexpr double default_regularization = 0.0001;

    for (sz c = 0;
         c < ClassCount;
         ++c) {

        double score =
            bias_[c];

        for (sz i = 0;
             i < feature_count_;
             ++i) {

            score +=
                features[i] *
                effective_weight(
                    c,
                    i,
                    default_learning_rate,
                    default_regularization
                );
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

Prediction LinearSVM::predict(
    const SparseVector& features
) const noexcept {

    Prediction prediction;

    if (!trained_ ||
        !features.valid()) {
        return prediction;
    }

    constexpr double learning_rate = 0.01;
    constexpr double regularization = 0.0001;

    arr<double, ClassCount> scores{};

    for (sz c = 0;
         c < ClassCount;
         ++c) {

        double score =
            bias_[c];

        for (sz i = 0;
             i < features.indices.size();
             ++i) {

            const sz index =
                features.indices[i];

            if (index >= feature_count_) {
                continue;
            }

            score +=
                features.values[i] *
                effective_weight(
                    c,
                    index,
                    learning_rate,
                    regularization
                );
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

const vec<vec<double>>&
LinearSVM::weights() const noexcept {
    return weights_;
}

const arr<double, 3>&
LinearSVM::bias() const noexcept {
    return bias_;
}

bool LinearSVM::trained() const noexcept {
    return trained_;
}

} // namespace sentiment
