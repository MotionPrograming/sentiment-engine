#include "sentiment/ml/evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sentiment {

EvaluationResult Evaluator::evaluate(
    const vec<Sentiment>& actual,
    const vec<Sentiment>& predicted
) {

    EvaluationResult result{};

    if (
        actual.empty() ||
        actual.size() != predicted.size()
    ) {
        return result;
    }

    constexpr sz class_count = 3;
    u64 correct = 0;

    for (sz i = 0; i < actual.size(); ++i) {

        const sz actual_class =
            static_cast<sz>(actual[i]);

        const sz predicted_class =
            static_cast<sz>(predicted[i]);

        if (
            actual_class >= class_count ||
            predicted_class >= class_count
        ) {
            continue;
        }

        ++result.confusion[actual_class][predicted_class];

        if (actual_class == predicted_class) {
            ++correct;
        }
    }

    result.total_samples = actual.size();
    result.correct_predictions = correct;

    result.accuracy =
        static_cast<double>(correct) /
        static_cast<double>(actual.size());

    u64 total_support = 0;
    double weighted_f1_sum = 0.0;

    for (sz c = 0; c < class_count; ++c) {

        const u64 tp = result.confusion[c][c];
        u64 fp = 0;
        u64 fn = 0;
        u64 support = 0;

        for (sz i = 0; i < class_count; ++i) {

            support += result.confusion[c][i];

            if (i != c) {
                fn += result.confusion[c][i];
                fp += result.confusion[i][c];
            }
        }

        const double precision =
            (tp + fp > 0)
                ? static_cast<double>(tp) / static_cast<double>(tp + fp)
                : 0.0;

        const double recall =
            (tp + fn > 0)
                ? static_cast<double>(tp) / static_cast<double>(tp + fn)
                : 0.0;

        const double f1 =
            (precision + recall > 0.0)
                ? 2.0 * precision * recall / (precision + recall)
                : 0.0;

        result.precision[c] = precision;
        result.recall[c]    = recall;
        result.f1[c]        = f1;

        result.macro_f1 += f1;
        total_support  += support;

        weighted_f1_sum += f1 * static_cast<double>(support);
    }

    result.macro_f1 /= static_cast<double>(class_count);

    if (total_support > 0) {
        result.weighted_f1 =
            weighted_f1_sum / static_cast<double>(total_support);
    }

    return result;
}

} // namespace sentiment