#include "sentiment/ml/linear_svm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std;

namespace sentiment {

/*
 * ============================================================
 * File format
 * ============================================================
 *
 * Magic:
 *
 *     SENTSVMD5
 *
 * Stores:
 *
 *     feature_count
 *     training_steps
 *     margin_violations
 *     trained
 *     class weights
 *     biases
 *     model weights
 *
 * ============================================================
 */

namespace {

constexpr char ModelMagic[] = "SENTSVMD5";

bool finite_positive(double value)
{
    return isfinite(value) &&
           value > 0.0;
}

bool finite_non_negative(double value)
{
    return isfinite(value) &&
           value >= 0.0;
}

} // namespace


/*
 * ============================================================
 * Constructor
 * ============================================================
 */

LinearSVM::LinearSVM(
    size_t feature_count)
    : feature_count_(feature_count)
{
    for (auto& class_weights : weights_) {

        class_weights.assign(
            feature_count_,
            0.0
        );
    }

    bias_.fill(0.0);

    class_weights_.fill(1.0);

    trained_ = false;
    training_steps_ = 0;
    margin_violations_ = 0;
}


/*
 * ============================================================
 * Class weights
 * ============================================================
 */

void LinearSVM::set_class_weights(
    const array<double, ClassCount>& weights)
{
    for (size_t c = 0; c < ClassCount; ++c) {

        const double value = weights[c];

        if (!finite_positive(value)) {

            throw invalid_argument(
                "LinearSVM: class weight must be "
                "finite and positive."
            );
        }

        /*
         * Clamp only to the production-safe range.
         *
         * IMPORTANT:
         *
         * 4.5 remains 4.5.
         */
        class_weights_[c] =
            clamp(
                value,
                MinimumClassWeight,
                MaximumClassWeight
            );
    }
}


const array<double, LinearSVM::ClassCount>&
LinearSVM::class_weights() const noexcept
{
    return class_weights_;
}


/*
 * ============================================================
 * Decision scores
 * ============================================================
 *
 * Sparse dot product:
 *
 *     score[c] =
 *
 *         bias[c]
 *         +
 *         sum(
 *             weight[c][index] *
 *             feature_value
 *         )
 *
 * Uses raw pointers for the hot loop.
 * ============================================================
 */

array<double, LinearSVM::ClassCount>
LinearSVM::decision_scores(
    const SparseVector& features) const
{
    array<double, ClassCount> scores{
        bias_[0],
        bias_[1],
        bias_[2]
    };

    if (features.empty()) {
        return scores;
    }

    const size_t count =
        features.indices.size();

    if (count != features.values.size()) {

        throw invalid_argument(
            "LinearSVM: invalid SparseVector."
        );
    }

    const auto* index_ptr =
        features.indices.data();

    const auto* value_ptr =
        features.values.data();

    const auto* w0 =
        weights_[0].data();

    const auto* w1 =
        weights_[1].data();

    const auto* w2 =
        weights_[2].data();

    double score0 = scores[0];
    double score1 = scores[1];
    double score2 = scores[2];

    for (size_t i = 0; i < count; ++i) {

        const uint32_t index =
            index_ptr[i];

        if (index >= feature_count_) {
            continue;
        }

        const double value =
            value_ptr[i];

        score0 +=
            w0[index] * value;

        score1 +=
            w1[index] * value;

        score2 +=
            w2[index] * value;
    }

    scores[0] = score0;
    scores[1] = score1;
    scores[2] = score2;

    return scores;
}


/*
 * ============================================================
 * train_sample
 * ============================================================
 *
 * Crammer-Singer style multiclass hinge loss:
 *
 * For true class y:
 *
 *     margin = score[y] - score[c]
 *
 * Violation:
 *
 *     margin < 1
 *
 * For every violating class:
 *
 *     W[y] += learning_rate * weight * x
 *     W[c] -= learning_rate * weight * x
 *
 * Class weight is applied directly to the gradient.
 *
 * Neutral = 4.5 therefore receives approximately 4.5x
 * update magnitude compared with a class having weight 1.0,
 * subject to the safety cap.
 *
 * ============================================================
 */

void LinearSVM::train_sample(
    const SparseVector& features,
    Sentiment label,
    double learning_rate,
    double regularization)
{
    if (feature_count_ == 0) {

        throw logic_error(
            "LinearSVM: cannot train a zero-feature model."
        );
    }

    if (!finite_positive(learning_rate)) {

        throw invalid_argument(
            "LinearSVM: learning rate must be "
            "finite and positive."
        );
    }

    if (!finite_non_negative(regularization)) {

        throw invalid_argument(
            "LinearSVM: regularization must be "
            "finite and non-negative."
        );
    }

    if (features.indices.size() !=
        features.values.size()) {

        throw invalid_argument(
            "LinearSVM: SparseVector indices/value "
            "size mismatch."
        );
    }

    const size_t true_class =
        static_cast<size_t>(label);

    if (true_class >= ClassCount) {

        throw invalid_argument(
            "LinearSVM: invalid sentiment label."
        );
    }

    /*
     * --------------------------------------------------------
     * Calculate all class scores.
     * --------------------------------------------------------
     */

    double score0 = bias_[0];
    double score1 = bias_[1];
    double score2 = bias_[2];

    const auto* index_ptr =
        features.indices.data();

    const auto* value_ptr =
        features.values.data();

    const auto* w0 =
        weights_[0].data();

    const auto* w1 =
        weights_[1].data();

    const auto* w2 =
        weights_[2].data();

    const size_t feature_count =
        features.indices.size();

    for (size_t i = 0;
         i < feature_count;
         ++i) {

        const uint32_t index =
            index_ptr[i];

        if (index >= feature_count_) {
            continue;
        }

        const double value =
            value_ptr[i];

        score0 +=
            w0[index] * value;

        score1 +=
            w1[index] * value;

        score2 +=
            w2[index] * value;
    }

    const array<double, ClassCount> scores{
        score0,
        score1,
        score2
    };

    const double true_score =
        scores[true_class];

    /*
     * --------------------------------------------------------
     * Find violating classes.
     * --------------------------------------------------------
     */

    array<bool, ClassCount> violating{
        false,
        false,
        false
    };

    size_t violation_count = 0;

    for (size_t c = 0;
         c < ClassCount;
         ++c) {

        if (c == true_class) {
            continue;
        }

        const double margin =
            true_score -
            scores[c];

        if (margin < Margin) {

            violating[c] = true;

            ++violation_count;
        }
    }

    ++training_steps_;

    /*
     * No hinge violation:
     *
     * Only regularization is not applied here because this
     * implementation uses sparse/touched-feature decay to
     * keep the hot path efficient.
     */
    if (violation_count == 0) {

        trained_ = true;

        return;
    }

    margin_violations_ +=
        violation_count;

    /*
     * --------------------------------------------------------
     * Class-weighted learning rate.
     * --------------------------------------------------------
     *
     * THIS is where the 4.5 Neutral weight enters the gradient.
     *
     * Example:
     *
     * LR = 0.001
     * Neutral weight = 4.5
     * violations = 1
     *
     * raw scale = 0.0045
     *
     * capped at MaximumGradientScale = 0.0050
     *
     * Therefore 4.5 is not silently reduced to 2.0.
     */

    const double sample_weight =
        class_weights_[true_class];

    const double raw_scale =
        learning_rate *
        sample_weight /
        static_cast<double>(violation_count);

    const double update_scale =
        min(
            raw_scale,
            MaximumGradientScale
        );

    /*
     * --------------------------------------------------------
     * L2 decay
     * --------------------------------------------------------
     *
     * Decay:
     *
     *     w = w * (1 - lr * lambda)
     *
     * Clamped to prevent numerical problems.
     */

    const double raw_decay =
        1.0 -
        learning_rate *
        regularization;

    const double decay =
        clamp(
            raw_decay,
            0.0,
            1.0
        );

    /*
     * --------------------------------------------------------
     * Sparse feature update
     * --------------------------------------------------------
     */

    auto* mutable_index_ptr =
        features.indices.data();

    const auto* mutable_value_ptr =
        features.values.data();

    auto* true_weights =
        weights_[true_class].data();

    /*
     * First decay touched features belonging to the true
     * class and violating classes.
     *
     * Then apply the hinge gradient.
     */
    for (size_t i = 0;
         i < feature_count;
         ++i) {

        const uint32_t index =
            mutable_index_ptr[i];

        if (index >= feature_count_) {
            continue;
        }

        const double value =
            mutable_value_ptr[i];

        /*
         * Ignore pathological non-finite feature values.
         */
        if (!isfinite(value)) {
            continue;
        }

        /*
         * True class:
         *
         *     + update
         */
        true_weights[index] *= decay;

        true_weights[index] +=
            update_scale * value;

        /*
         * Violating classes:
         *
         *     - update
         */
        for (size_t c = 0;
             c < ClassCount;
             ++c) {

            if (!violating[c]) {
                continue;
            }

            if (c == true_class) {
                continue;
            }

            auto* class_weight_vector =
                weights_[c].data();

            class_weight_vector[index] *= decay;

            class_weight_vector[index] -=
                update_scale * value;
        }
    }

    /*
     * --------------------------------------------------------
     * Bias update
     * --------------------------------------------------------
     *
     * Feature gradients use the full update_scale.
     *
     * Bias uses a smaller scale for stability.
     */

    const double bias_delta =
        update_scale *
        BiasLearningRateScale;

    bias_[true_class] +=
        bias_delta;

    for (size_t c = 0;
         c < ClassCount;
         ++c) {

        if (!violating[c]) {
            continue;
        }

        if (c == true_class) {
            continue;
        }

        bias_[c] -=
            bias_delta;
    }

    trained_ = true;
}


/*
 * ============================================================
 * Prediction
 * ============================================================
 */

Sentiment LinearSVM::predict(
    const SparseVector& features) const
{
    const auto scores =
        decision_scores(features);

    size_t best_class = 0;

    if (scores[1] > scores[best_class]) {
        best_class = 1;
    }

    if (scores[2] > scores[best_class]) {
        best_class = 2;
    }

    return static_cast<Sentiment>(
        best_class
    );
}


/*
 * ============================================================
 * Confidence
 * ============================================================
 *
 * Numerically stable softmax:
 *
 *     exp(score - max_score)
 *
 * Then:
 *
 *     confidence =
 *         max_probability
 *
 * This is NOT a calibrated probability.
 * It is a score-based confidence estimate.
 * ============================================================
 */

double LinearSVM::confidence(
    const SparseVector& features) const
{
    const auto scores =
        decision_scores(features);

    const double max_score =
        *max_element(
            scores.begin(),
            scores.end()
        );

    double denominator = 0.0;

    array<double, ClassCount> probabilities{};

    for (size_t c = 0;
         c < ClassCount;
         ++c) {

        const double value =
            exp(
                clamp(
                    scores[c] - max_score,
                    -50.0,
                    50.0
                )
            );

        probabilities[c] =
            value;

        denominator +=
            value;
    }

    if (!(denominator > 0.0) ||
        !isfinite(denominator)) {

        return 0.0;
    }

    size_t best_class = 0;

    for (size_t c = 1;
         c < ClassCount;
         ++c) {

        if (probabilities[c] >
            probabilities[best_class]) {

            best_class = c;
        }
    }

    return probabilities[best_class] /
           denominator;
}


/*
 * ============================================================
 * Model information
 * ============================================================
 */

size_t LinearSVM::feature_count() const noexcept
{
    return feature_count_;
}


bool LinearSVM::trained() const noexcept
{
    return trained_;
}


uint64_t LinearSVM::training_steps() const noexcept
{
    return training_steps_;
}


uint64_t LinearSVM::margin_violations() const noexcept
{
    return margin_violations_;
}


/*
 * ============================================================
 * SAVE
 * ============================================================
 */

void LinearSVM::save(
    const string& path) const
{
    ofstream output(
        path,
        ios::binary | ios::trunc
    );

    if (!output) {

        throw runtime_error(
            "LinearSVM: cannot open model for writing: " +
            path
        );
    }

    /*
     * Magic
     */
    output.write(
        ModelMagic,
        sizeof(ModelMagic) - 1
    );

    /*
     * Version
     */
    const uint32_t version = 5;

    output.write(
        reinterpret_cast<const char*>(&version),
        sizeof(version)
    );

    /*
     * Feature count
     */
    const uint64_t feature_count =
        static_cast<uint64_t>(
            feature_count_
        );

    output.write(
        reinterpret_cast<const char*>(&feature_count),
        sizeof(feature_count)
    );

    /*
     * Training state
     */
    const uint8_t trained =
        trained_ ? 1 : 0;

    output.write(
        reinterpret_cast<const char*>(&trained),
        sizeof(trained)
    );

    output.write(
        reinterpret_cast<const char*>(&training_steps_),
        sizeof(training_steps_)
    );

    output.write(
        reinterpret_cast<const char*>(&margin_violations_),
        sizeof(margin_violations_)
    );

    /*
     * Class weights
     */
    output.write(
        reinterpret_cast<const char*>(
            class_weights_.data()
        ),
        sizeof(double) * ClassCount
    );

    /*
     * Biases
     */
    output.write(
        reinterpret_cast<const char*>(
            bias_.data()
        ),
        sizeof(double) * ClassCount
    );

    /*
     * Weight matrices
     */
    for (size_t c = 0;
         c < ClassCount;
         ++c) {

        output.write(
            reinterpret_cast<const char*>(
                weights_[c].data()
            ),
            sizeof(double) *
            weights_[c].size()
        );
    }

    if (!output) {

        throw runtime_error(
            "LinearSVM: failed while writing model: " +
            path
        );
    }
}


/*
 * ============================================================
 * LOAD
 * ============================================================
 */

void LinearSVM::load(
    const string& path)
{
    ifstream input(
        path,
        ios::binary
    );

    if (!input) {

        throw runtime_error(
            "LinearSVM: cannot open model: " +
            path
        );
    }

    /*
     * Magic
     */
    char magic[
        sizeof(ModelMagic) - 1
    ]{};

    input.read(
        magic,
        sizeof(magic)
    );

    if (!input ||
        !equal(
            begin(magic),
            end(magic),
            begin(ModelMagic)
        )) {

        throw runtime_error(
            "LinearSVM: invalid model file."
        );
    }

    /*
     * Version
     */
    uint32_t version = 0;

    input.read(
        reinterpret_cast<char*>(&version),
        sizeof(version)
    );

    if (!input) {

        throw runtime_error(
            "LinearSVM: invalid model version."
        );
    }

    if (version != 5) {

        throw runtime_error(
            "LinearSVM: unsupported model version."
        );
    }

    /*
     * Feature count
     */
    uint64_t stored_feature_count = 0;

    input.read(
        reinterpret_cast<char*>(&stored_feature_count),
        sizeof(stored_feature_count)
    );

    if (!input) {

        throw runtime_error(
            "LinearSVM: invalid feature count."
        );
    }

    if (stored_feature_count >
        static_cast<uint64_t>(
            numeric_limits<size_t>::max()
        )) {

        throw runtime_error(
            "LinearSVM: feature count overflow."
        );
    }

    feature_count_ =
        static_cast<size_t>(
            stored_feature_count
        );

    /*
     * Allocate model weights.
     */
    for (auto& class_weight_vector : weights_) {

        class_weight_vector.assign(
            feature_count_,
            0.0
        );
    }

    /*
     * Trained flag
     */
    uint8_t trained = 0;

    input.read(
        reinterpret_cast<char*>(&trained),
        sizeof(trained)
    );

    /*
     * Counters
     */
    input.read(
        reinterpret_cast<char*>(&training_steps_),
        sizeof(training_steps_)
    );

    input.read(
        reinterpret_cast<char*>(&margin_violations_),
        sizeof(margin_violations_)
    );

    /*
     * Class weights
     */
    input.read(
        reinterpret_cast<char*>(
            class_weights_.data()
        ),
        sizeof(double) * ClassCount
    );

    /*
     * Bias
     */
    input.read(
        reinterpret_cast<char*>(
            bias_.data()
        ),
        sizeof(double) * ClassCount
    );

    /*
     * Weight matrices
     */
    for (size_t c = 0;
         c < ClassCount;
         ++c) {

        input.read(
            reinterpret_cast<char*>(
                weights_[c].data()
            ),
            sizeof(double) *
            weights_[c].size()
        );
    }

    if (!input) {

        throw runtime_error(
            "LinearSVM: corrupted model file."
        );
    }

    /*
     * Validate loaded class weights.
     */
    for (double weight : class_weights_) {

        if (!finite_positive(weight)) {

            throw runtime_error(
                "LinearSVM: invalid class weight "
                "inside model."
            );
        }

        if (weight < MinimumClassWeight ||
            weight > MaximumClassWeight) {

            throw runtime_error(
                "LinearSVM: class weight outside "
                "supported range."
            );
        }
    }

    /*
     * Validate biases.
     */
    for (double value : bias_) {

        if (!isfinite(value)) {

            throw runtime_error(
                "LinearSVM: invalid bias in model."
            );
        }
    }

    /*
     * Validate weights.
     */
    for (const auto& class_weight_vector :
         weights_) {

        for (double value :
             class_weight_vector) {

            if (!isfinite(value)) {

                throw runtime_error(
                    "LinearSVM: invalid model weight."
                );
            }
        }
    }

    trained_ =
        (trained != 0);
}

} // namespace sentiment