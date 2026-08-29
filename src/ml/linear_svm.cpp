#include "sentiment/ml/linear_svm.hpp"
#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

LinearSVM::LinearSVM(
    sz feature_count
)
    : feature_count_(feature_count),
      weights_(
          ClassCount,
          vec<double>(
              feature_count,
              0.0
          )
      ) {
}

void LinearSVM::set_weights(
    sz class_id,
    vec<double> weights
) {
    if (class_id >= ClassCount) {
        return;
    }

    if (weights.size() != feature_count_) {
        return;
    }

    weights_[class_id] = move(weights);
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

    for (sz c = 0; c < ClassCount; ++c) {

        double score = bias_[c];
        const auto& weights = weights_[c];

        for (sz i = 0; i < feature_count_; ++i) {
            score += features[i] * weights[i];
        }

        scores[c] = score;
    }

    const auto best = static_cast<sz>(
        distance(
            scores.begin(),
            max_element(scores.begin(), scores.end())
        )
    );

    prediction.label = static_cast<Sentiment>(best);
    prediction.score = scores[best];

    return prediction;
}

} // namespace sentiment