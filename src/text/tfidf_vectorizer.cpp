#include "sentiment/text/tfidf_vectorizer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sentiment {

TfidfVectorizer::TfidfVectorizer(
    sz max_features
)
    : max_features_(max_features) {
}

void TfidfVectorizer::fit(
    const vec<str>& documents
) {
    vocabulary_ = Vocabulary{};
    idf_.clear();
    fitted_ = false;

    if (
        documents.empty() ||
        max_features_ == 0
    ) {
        return;
    }

    const sz document_count =
        documents.size();

    std::unordered_map<str, sz>
        document_frequency;

    for (const auto& document : documents) {

        const auto tokens =
            tokenizer_.tokenize(document);

        std::unordered_set<str>
            unique_tokens;

        unique_tokens.reserve(
            tokens.size()
        );

        for (const auto& token : tokens) {
            unique_tokens.insert(token);
        }

        for (const auto& token : unique_tokens) {
            ++document_frequency[token];
        }
    }

    vec<std::pair<str, sz>> terms;

    terms.reserve(
        document_frequency.size()
    );

    for (const auto& [term, df] :
         document_frequency) {

        terms.emplace_back(
            term,
            df
        );
    }

    std::sort(
        terms.begin(),
        terms.end(),
        [](const auto& lhs,
           const auto& rhs) {

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

    for (sz i = 0;
         i < feature_count;
         ++i) {

        const auto& [term, df] =
            terms[i];

        vocabulary_.add(term);

        /*
         * Smoothed IDF:
         *
         * log((N + 1) / (df + 1)) + 1
         */
        idf_[i] =
            std::log(
                (1.0 +
                 static_cast<double>(
                     document_count
                 )) /
                (1.0 +
                 static_cast<double>(
                     df
                 ))
            ) + 1.0;
    }

    fitted_ = true;
}

void TfidfVectorizer::fit_stream(
    sz document_count,
    const std::function<bool(str&)>& next_document
) {
    vocabulary_ = Vocabulary{};
    idf_.clear();
    fitted_ = false;

    if (
        document_count == 0 ||
        max_features_ == 0 ||
        !next_document
    ) {
        return;
    }

    std::unordered_map<str, sz>
        document_frequency;

    str document;

    for (sz document_index = 0;
         document_index < document_count;
         ++document_index) {

        document.clear();

        if (!next_document(document)) {
            break;
        }

        const auto tokens =
            tokenizer_.tokenize(document);

        std::unordered_set<str>
            unique_tokens;

        unique_tokens.reserve(
            tokens.size()
        );

        for (const auto& token : tokens) {
            unique_tokens.insert(token);
        }

        for (const auto& token : unique_tokens) {
            ++document_frequency[token];
        }
    }

    vec<std::pair<str, sz>> terms;

    terms.reserve(
        document_frequency.size()
    );

    for (const auto& [term, df] :
         document_frequency) {

        terms.emplace_back(
            term,
            df
        );
    }

    std::sort(
        terms.begin(),
        terms.end(),
        [](const auto& lhs,
           const auto& rhs) {

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

    for (sz i = 0;
         i < feature_count;
         ++i) {

        const auto& [term, df] =
            terms[i];

        vocabulary_.add(term);

        idf_[i] =
            std::log(
                (1.0 +
                 static_cast<double>(
                     document_count
                 )) /
                (1.0 +
                 static_cast<double>(
                     df
                 ))
            ) + 1.0;
    }

    fitted_ = feature_count > 0;
}

vec<double>
TfidfVectorizer::transform(
    sv document
) const {

    vec<double> result(
        vocabulary_.size(),
        0.0
    );

    if (
        !fitted_ ||
        vocabulary_.size() == 0
    ) {
        return result;
    }

    const auto tokens =
        tokenizer_.tokenize(document);

    for (const auto& token : tokens) {

        const auto id =
            vocabulary_.find(token);

        if (
            id != Vocabulary::InvalidTokenId
        ) {
            ++result[id];
        }
    }

    for (sz i = 0;
         i < result.size();
         ++i) {

        if (result[i] > 0.0) {

            result[i] *= idf_[i];
        }
    }

    double squared_norm = 0.0;

    for (const auto value : result) {
        squared_norm +=
            value * value;
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

SparseVector
TfidfVectorizer::transform_sparse(
    sv document
) const {

    SparseVector result;

    if (
        !fitted_ ||
        vocabulary_.size() == 0
    ) {
        return result;
    }

    const auto tokens =
        tokenizer_.tokenize(document);

    /*
     * TF counts only active vocabulary
     * features.
     */
    std::unordered_map<sz, double>
        term_frequency;

    term_frequency.reserve(
        tokens.size()
    );

    for (const auto& token : tokens) {

        const auto id =
            vocabulary_.find(token);

        if (
            id != Vocabulary::InvalidTokenId
        ) {
            ++term_frequency[id];
        }
    }

    if (term_frequency.empty()) {
        return result;
    }

    /*
     * Calculate unnormalized TF-IDF first.
     */
    result.reserve(
        term_frequency.size()
    );

    double squared_norm = 0.0;

    for (const auto& [id, tf] :
         term_frequency) {

        if (id >= idf_.size()) {
            continue;
        }

        const double value =
            tf * idf_[id];

        if (value == 0.0) {
            continue;
        }

        result.indices.push_back(id);
        result.values.push_back(value);

        squared_norm +=
            value * value;
    }

    /*
     * L2 normalization.
     */
    if (squared_norm > 0.0) {

        const double norm =
            std::sqrt(squared_norm);

        for (auto& value :
             result.values) {

            value /= norm;
        }
    }

    return result;
}

const Vocabulary&
TfidfVectorizer::vocabulary()
    const noexcept {

    return vocabulary_;
}

const vec<double>&
TfidfVectorizer::idf()
    const noexcept {

    return idf_;
}

bool TfidfVectorizer::set_model_data(
    Vocabulary vocabulary,
    vec<double> idf
) {

    if (
        vocabulary.size() == 0 ||
        vocabulary.size() != idf.size()
    ) {

        fitted_ = false;

        return false;
    }

    vocabulary_ =
        std::move(vocabulary);

    idf_ =
        std::move(idf);

    fitted_ = true;

    return true;
}

sz TfidfVectorizer::vocabulary_size()
    const noexcept {

    return vocabulary_.size();
}

} // namespace sentiment
