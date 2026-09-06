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

namespace {

constexpr char ModelMagic[] = "SENTSVMD5";

bool finite_positive(double value) { return isfinite(value) && value > 0.0; }
bool finite_non_negative(double value) { return isfinite(value) && value >= 0.0; }

} // namespace

LinearSVM::LinearSVM(size_t feature_count) : feature_count_(feature_count) {
    for (auto& class_weights : weights_) class_weights.assign(feature_count_, 0.0);
    bias_.fill(0.0);
    class_weights_.fill(1.0);
    trained_ = false;
    training_steps_ = 0;
    margin_violations_ = 0;
}

void LinearSVM::set_class_weights(const array<double, ClassCount>& weights) {
    for (size_t c = 0; c < ClassCount; ++c) {
        const double value = weights[c];
        if (!finite_positive(value)) {
            throw invalid_argument("LinearSVM: class weight must be finite and positive.");
        }
        class_weights_[c] = clamp(value, MinimumClassWeight, MaximumClassWeight);
    }
}

const array<double, LinearSVM::ClassCount>& LinearSVM::class_weights() const noexcept {
    return class_weights_;
}

array<double, LinearSVM::ClassCount> LinearSVM::decision_scores(const SparseVector& features) const {
    array<double, ClassCount> scores{bias_[0], bias_[1], bias_[2]};
    if (features.empty()) return scores;

    const size_t count = features.indices.size();
    if (count != features.values.size()) {
        throw invalid_argument("LinearSVM: invalid SparseVector.");
    }

    const auto* index_ptr = features.indices.data();
    const auto* value_ptr = features.values.data();
    const auto* w0 = weights_[0].data();
    const auto* w1 = weights_[1].data();
    const auto* w2 = weights_[2].data();

    double score0 = scores[0], score1 = scores[1], score2 = scores[2];

    for (size_t i = 0; i < count; ++i) {
        const uint32_t index = index_ptr[i];
        if (index >= feature_count_) continue;

        const double value = value_ptr[i];
        score0 += w0[index] * value;
        score1 += w1[index] * value;
        score2 += w2[index] * value;
    }

    scores[0] = score0;
    scores[1] = score1;
    scores[2] = score2;
    return scores;
}

void LinearSVM::train_sample(const SparseVector& features, Sentiment label, double learning_rate, double regularization) {
    if (feature_count_ == 0) throw logic_error("LinearSVM: cannot train a zero-feature model.");
    if (!finite_positive(learning_rate)) throw invalid_argument("LinearSVM: learning rate must be finite and positive.");
    if (!finite_non_negative(regularization)) throw invalid_argument("LinearSVM: regularization must be finite and non-negative.");
    if (features.indices.size() != features.values.size()) throw invalid_argument("LinearSVM: SparseVector indices/value size mismatch.");

    const size_t true_class = static_cast<size_t>(label);
    if (true_class >= ClassCount) throw invalid_argument("LinearSVM: invalid sentiment label.");

    double score0 = bias_[0], score1 = bias_[1], score2 = bias_[2];
    const auto* index_ptr = features.indices.data();
    const auto* value_ptr = features.values.data();
    const auto* w0 = weights_[0].data();
    const auto* w1 = weights_[1].data();
    const auto* w2 = weights_[2].data();
    const size_t feature_count = features.indices.size();

    for (size_t i = 0; i < feature_count; ++i) {
        const uint32_t index = index_ptr[i];
        if (index >= feature_count_) continue;

        const double value = value_ptr[i];
        score0 += w0[index] * value;
        score1 += w1[index] * value;
        score2 += w2[index] * value;
    }

    const array<double, ClassCount> scores{score0, score1, score2};
    const double true_score = scores[true_class];

    array<bool, ClassCount> violating{false, false, false};
    size_t violation_count = 0;

    for (size_t c = 0; c < ClassCount; ++c) {
        if (c == true_class) continue;
        if (true_score - scores[c] < Margin) {
            violating[c] = true;
            ++violation_count;
        }
    }

    ++training_steps_;
    if (violation_count == 0) {
        trained_ = true;
        return;
    }

    margin_violations_ += violation_count;

    const double sample_weight = class_weights_[true_class];
    const double raw_scale = learning_rate * sample_weight / static_cast<double>(violation_count);
    const double update_scale = min(raw_scale, MaximumGradientScale);
    const double decay = clamp(1.0 - learning_rate * regularization, 0.0, 1.0);

    auto* mutable_index_ptr = features.indices.data();
    const auto* mutable_value_ptr = features.values.data();
    auto* true_weights = weights_[true_class].data();

    for (size_t i = 0; i < feature_count; ++i) {
        const uint32_t index = mutable_index_ptr[i];
        if (index >= feature_count_) continue;

        const double value = mutable_value_ptr[i];
        if (!isfinite(value)) continue;

        true_weights[index] = true_weights[index] * decay + update_scale * value;

        for (size_t c = 0; c < ClassCount; ++c) {
            if (!violating[c] || c == true_class) continue;
            auto* class_weight_vector = weights_[c].data();
            class_weight_vector[index] = class_weight_vector[index] * decay - update_scale * value;
        }
    }

    const double bias_delta = update_scale * BiasLearningRateScale;
    bias_[true_class] += bias_delta;

    for (size_t c = 0; c < ClassCount; ++c) {
        if (!violating[c] || c == true_class) continue;
        bias_[c] -= bias_delta;
    }

    trained_ = true;
}

Sentiment LinearSVM::predict(const SparseVector& features) const {
    const auto scores = decision_scores(features);
    size_t best_class = 0;
    if (scores[1] > scores[best_class]) best_class = 1;
    if (scores[2] > scores[best_class]) best_class = 2;
    return static_cast<Sentiment>(best_class);
}

double LinearSVM::confidence(const SparseVector& features) const {
    const auto scores = decision_scores(features);
    const double max_score = *max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    array<double, ClassCount> probabilities{};

    for (size_t c = 0; c < ClassCount; ++c) {
        const double value = exp(clamp(scores[c] - max_score, -50.0, 50.0));
        probabilities[c] = value;
        denominator += value;
    }

    if (!(denominator > 0.0) || !isfinite(denominator)) return 0.0;

    size_t best_class = 0;
    for (size_t c = 1; c < ClassCount; ++c) {
        if (probabilities[c] > probabilities[best_class]) best_class = c;
    }

    return probabilities[best_class] / denominator;
}

size_t LinearSVM::feature_count() const noexcept { return feature_count_; }
bool LinearSVM::trained() const noexcept { return trained_; }
uint64_t LinearSVM::training_steps() const noexcept { return training_steps_; }
uint64_t LinearSVM::margin_violations() const noexcept { return margin_violations_; }

void LinearSVM::save(const string& path) const {
    ofstream output(path, ios::binary | ios::trunc);
    if (!output) throw runtime_error("LinearSVM: cannot open model for writing: " + path);

    output.write(ModelMagic, sizeof(ModelMagic) - 1);
    const uint32_t version = 5;
    output.write(reinterpret_cast<const char*>(&version), sizeof(version));

    const uint64_t feature_count = static_cast<uint64_t>(feature_count_);
    output.write(reinterpret_cast<const char*>(&feature_count), sizeof(feature_count));

    const uint8_t trained = trained_ ? 1 : 0;
    output.write(reinterpret_cast<const char*>(&trained), sizeof(trained));
    output.write(reinterpret_cast<const char*>(&training_steps_), sizeof(training_steps_));
    output.write(reinterpret_cast<const char*>(&margin_violations_), sizeof(margin_violations_));

    output.write(reinterpret_cast<const char*>(class_weights_.data()), sizeof(double) * ClassCount);
    output.write(reinterpret_cast<const char*>(bias_.data()), sizeof(double) * ClassCount);

    for (size_t c = 0; c < ClassCount; ++c) {
        output.write(reinterpret_cast<const char*>(weights_[c].data()), sizeof(double) * weights_[c].size());
    }

    if (!output) throw runtime_error("LinearSVM: failed while writing model: " + path);
}

void LinearSVM::load(const string& path) {
    ifstream input(path, ios::binary);
    if (!input) throw runtime_error("LinearSVM: cannot open model: " + path);

    char magic[sizeof(ModelMagic) - 1]{};
    input.read(magic, sizeof(magic));
    if (!input || !equal(begin(magic), end(magic), begin(ModelMagic))) {
        throw runtime_error("LinearSVM: invalid model file.");
    }

    uint32_t version = 0;
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!input || version != 5) throw runtime_error("LinearSVM: unsupported model version.");

    uint64_t stored_feature_count = 0;
    input.read(reinterpret_cast<char*>(&stored_feature_count), sizeof(stored_feature_count));
    if (!input || stored_feature_count > static_cast<uint64_t>(numeric_limits<size_t>::max())) {
        throw runtime_error("LinearSVM: invalid feature count.");
    }

    feature_count_ = static_cast<size_t>(stored_feature_count);
    for (auto& class_weight_vector : weights_) class_weight_vector.assign(feature_count_, 0.0);

    uint8_t trained = 0;
    input.read(reinterpret_cast<char*>(&trained), sizeof(trained));
    input.read(reinterpret_cast<char*>(&training_steps_), sizeof(training_steps_));
    input.read(reinterpret_cast<char*>(&margin_violations_), sizeof(margin_violations_));

    input.read(reinterpret_cast<char*>(class_weights_.data()), sizeof(double) * ClassCount);
    input.read(reinterpret_cast<char*>(bias_.data()), sizeof(double) * ClassCount);

    for (size_t c = 0; c < ClassCount; ++c) {
        input.read(reinterpret_cast<char*>(weights_[c].data()), sizeof(double) * weights_[c].size());
    }

    if (!input) throw runtime_error("LinearSVM: corrupted model file.");

    for (double weight : class_weights_) {
        if (!finite_positive(weight) || weight < MinimumClassWeight || weight > MaximumClassWeight) {
            throw runtime_error("LinearSVM: invalid class weight inside model.");
        }
    }

    for (double value : bias_) {
        if (!isfinite(value)) throw runtime_error("LinearSVM: invalid bias in model.");
    }

    for (const auto& class_weight_vector : weights_) {
        for (double value : class_weight_vector) {
            if (!isfinite(value)) throw runtime_error("LinearSVM: invalid model weight.");
        }
    }

    trained_ = (trained != 0);
}

} // namespace sentiment