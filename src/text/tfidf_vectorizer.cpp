#include "sentiment/text/tfidf_vectorizer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sentiment {

TfidfVectorizer::TfidfVectorizer(sz max_features)
    : max_features_(max_features) {
}

void TfidfVectorizer::fit(
    const vec<str>& documents
) {
    vocabulary_ = Vocabulary{};
    idf_.clear();
    fitted_ = false;

    if (documents.empty() || max_features_ == 0) {
        return;
    }

    const sz document_count = documents.size();

    std::unordered_map<str, sz> document_frequency;

    for (const auto& document : documents) {

        const auto tokens =
            tokenizer_.tokenize(document);

        std::unordered_set<str> unique_tokens;
        unique_tokens.reserve(tokens.size());

        for (const auto& token : tokens) {
            unique_tokens.insert(token);
        }

        for (const auto& token : unique_tokens) {
            ++document_frequency[token];
        }
    }

    vec<std::pair<str, sz>> terms;

    terms.reserve(document_frequency.size());

    for (const auto& [term, df] : document_frequency) {
        terms.emplace_back(term, df);
    }

    std::sort(
        terms.begin(),
        terms.end(),
        [](const auto& lhs, const auto& rhs) {

            if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
            }

            return lhs.first < rhs.first;
        }
    );

    const sz feature_count =
        std::min(
            max_features_,
            terms.size()
        );

    idf_.resize(feature_count);

    for (sz i = 0; i < feature_count; ++i) {

        const auto& [term, df] = terms[i];

        vocabulary_.add(term);

        /*
         * Smoothed IDF:
         *
         * idf = log((N + 1) / (df + 1)) + 1
         */
        idf_[i] =
            std::log(
                (1.0 + static_cast<double>(document_count)) /
                (1.0 + static_cast<double>(df))
            ) + 1.0;
    }

    fitted_ = true;
}

vec<double> TfidfVectorizer::transform(
    sv document
) const {
    vec<double> result(
        vocabulary_.size(),
        0.0
    );

    if (!fitted_ || vocabulary_.size() == 0) {
        return result;
    }

    const auto tokens =
        tokenizer_.tokenize(document);

    /*
     * Term frequency.
     */
    for (const auto& token : tokens) {

        const auto id =
            vocabulary_.find(token);

        if (id != static_cast<sz>(-1)) {
            ++result[id];
        }
    }

    /*
     * TF-IDF weighting.
     */
    for (sz i = 0; i < result.size(); ++i) {

        if (result[i] > 0.0) {
            result[i] *= idf_[i];
        }
    }

    /*
     * L2 normalization.
     */
    double squared_norm = 0.0;

    for (const auto value : result) {
        squared_norm += value * value;
    }

    if (squared_norm > 0.0) {

        const double norm =
            std::sqrt(squared_norm);

        for (auto& value : result) {
            value /= norm;
        }
    }

    return result;
}

sz TfidfVectorizer::vocabulary_size() const noexcept {
    return vocabulary_.size();
}

} // namespace sentiment
