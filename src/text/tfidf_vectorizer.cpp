#include "sentiment/text/tfidf_vectorizer.hpp"
using namespace std;

#include <algorithm>
using namespace std;
#include <cmath>
using namespace std;
#include <unordered_set>
using namespace std;

namespace sentiment {

TfidfVectorizer::TfidfVectorizer(
    std::size_t max_features
)
    : max_features_(max_features) {
}

void TfidfVectorizer::fit(
    const std::vector<std::string>& documents
) {

    const std::size_t document_count =
        documents.size();

    if (document_count == 0) {
        return;
    }

    std::unordered_map<std::string, std::size_t>
        document_frequency;

    for (const auto& document : documents) {

        auto tokens =
            tokenizer_.tokenize(document);

        std::unordered_set<std::string>
            unique_tokens;

        unique_tokens.reserve(tokens.size());

        for (const auto& token : tokens) {
            unique_tokens.insert(token);
        }

        for (const auto& token : unique_tokens) {

            ++document_frequency[token];
        }
    }

    std::vector<std::pair<std::string, std::size_t>>
        terms(
            document_frequency.begin(),
            document_frequency.end()
        );

    std::sort(
        terms.begin(),
        terms.end(),
        [](const auto& a, const auto& b) {

            return a.second > b.second;
        }
    );

    const std::size_t feature_count =
        std::min(
            max_features_,
            terms.size()
        );

    idf_.resize(feature_count);

    for (std::size_t i = 0; i < feature_count; ++i) {

        const auto& [term, df] = terms[i];

        const auto id =
            vocabulary_.add(term);

        (void)id;

        idf_[i] =
            std::log(
                (1.0 + static_cast<double>(document_count)) /
                (1.0 + static_cast<double>(df))
            ) + 1.0;
    }

    fitted_ = true;
}

std::vector<double>
TfidfVectorizer::transform(
    std::string_view document
) const {

    std::vector<double> result(
        vocabulary_.size(),
        0.0
    );

    if (!fitted_) {
        return result;
    }

    auto tokens =
        tokenizer_.tokenize(document);

    for (const auto& token : tokens) {

        const auto id =
            vocabulary_.find(token);

        if (id != static_cast<std::size_t>(-1) &&
            id < result.size()) {

            result[id] += 1.0;
        }
    }

    double norm = 0.0;

    for (std::size_t i = 0; i < result.size(); ++i) {

        if (result[i] > 0.0) {

            result[i] *= idf_[i];

            norm += result[i] * result[i];
        }
    }

    if (norm > 0.0) {

        norm = std::sqrt(norm);

        for (auto& value : result) {
            value /= norm;
        }
    }

    return result;
}

std::size_t
TfidfVectorizer::vocabulary_size() const noexcept {

    return vocabulary_.size();
}

} // namespace sentiment
