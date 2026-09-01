#include "sentiment/text/tfidf_vectorizer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace std;

namespace sentiment {

// ============================================================
// Constructor
// ============================================================

TfidfVectorizer::TfidfVectorizer(
    sz max_features
)
    : max_features_(
          max_features
      ) {
}


// ============================================================
// Collect unigram + bigram IDs
// ============================================================

void TfidfVectorizer::collect_term_ids(
    sv document,
    vec<u32>& ids
) const {

    ids.clear();

    if (
        document.empty() ||
        !fitted_ ||
        vocabulary_.size() == 0
    ) {
        return;
    }

    const auto tokens =
        tokenizer_.tokenize(document);

    if (tokens.empty())
        return;

    /*
     * Reserve enough for:
     *
     *     unigrams + bigrams
     */
    ids.reserve(
        tokens.size() * 2
    );

    /*
     * --------------------------------------------------------
     * Unigrams
     * --------------------------------------------------------
     */

    for (
        const string& token :
        tokens
    ) {

        if (token.empty())
            continue;

        const auto id =
            vocabulary_.find(token);

        if (
            id !=
            Vocabulary::InvalidId
        ) {

            ids.push_back(
                static_cast<u32>(id)
            );
        }
    }

    /*
     * --------------------------------------------------------
     * Bigrams
     * --------------------------------------------------------
     *
     * Same representation used during training:
     *
     * token1_token2
     * --------------------------------------------------------
     */

    if (tokens.size() >= 2) {

        string bigram;

        for (
            sz i = 0;
            i + 1 < tokens.size();
            ++i
        ) {

            if (
                tokens[i].empty() ||
                tokens[i + 1].empty()
            ) {
                continue;
            }

            bigram.clear();

            bigram.reserve(
                tokens[i].size() +
                tokens[i + 1].size() +
                1
            );

            bigram.append(
                tokens[i]
            );

            bigram.push_back('_');

            bigram.append(
                tokens[i + 1]
            );

            const auto id =
                vocabulary_.find(
                    bigram
                );

            if (
                id !=
                Vocabulary::InvalidId
            ) {

                ids.push_back(
                    static_cast<u32>(id)
                );
            }
        }
    }
}


// ============================================================
// Fit
// ============================================================

void TfidfVectorizer::fit(
    const vec<str>& documents
) {

    vocabulary_ =
        Vocabulary{};

    idf_.clear();

    fitted_ =
        false;

    if (
        documents.empty() ||
        max_features_ == 0
    ) {
        return;
    }

    unordered_map<
        string,
        u64
    > document_frequency;

    document_frequency.reserve(
        max_features_ * 4
    );

    u64 document_count =
        0;

    /*
     * Reusable tokenizer-term set.
     *
     * Avoids allocating a new hash table for every
     * document.
     */
    unordered_set<string> terms;

    terms.reserve(256);

    for (
        const string& document :
        documents
    ) {

        const auto tokens =
            tokenizer_.tokenize(
                document
            );

        if (tokens.empty())
            continue;

        terms.clear();

        /*
         * ----------------------------------------------------
         * Unigrams
         * ----------------------------------------------------
         */

        for (
            const string& token :
            tokens
        ) {

            if (!token.empty())
                terms.emplace(token);
        }

        /*
         * ----------------------------------------------------
         * Bigrams
         * ----------------------------------------------------
         */

        string bigram;

        if (tokens.size() >= 2) {

            for (
                sz i = 0;
                i + 1 < tokens.size();
                ++i
            ) {

                if (
                    tokens[i].empty() ||
                    tokens[i + 1].empty()
                ) {
                    continue;
                }

                bigram.clear();

                bigram.reserve(
                    tokens[i].size() +
                    tokens[i + 1].size() +
                    1
                );

                bigram.append(
                    tokens[i]
                );

                bigram.push_back('_');

                bigram.append(
                    tokens[i + 1]
                );

                terms.emplace(
                    bigram
                );
            }
        }

        ++document_count;

        /*
         * Each term contributes at most once per document.
         */
        for (
            const string& term :
            terms
        ) {

            ++document_frequency[term];
        }
    }

    if (document_count == 0)
        return;

    /*
     * --------------------------------------------------------
     * Feature selection
     * --------------------------------------------------------
     */

    vec<pair<string, u64>>
        terms_sorted;

    terms_sorted.reserve(
        document_frequency.size()
    );

    for (
        auto& entry :
        document_frequency
    ) {

        terms_sorted.emplace_back(
            std::move(entry.first),
            entry.second
        );
    }

    sort(
        terms_sorted.begin(),
        terms_sorted.end(),
        [](
            const auto& lhs,
            const auto& rhs
        ) {

            if (
                lhs.second !=
                rhs.second
            ) {

                return lhs.second >
                       rhs.second;
            }

            return lhs.first <
                   rhs.first;
        }
    );

    /*
     * --------------------------------------------------------
     * Minimum DF
     *
     * Keep consistent with train.cpp.
     * --------------------------------------------------------
     */

    constexpr u64 MinimumDocumentFrequency = 3;

    sz feature_count = 0;

    for (
        const auto& [term, df] :
        terms_sorted
    ) {

        if (
            df < MinimumDocumentFrequency
        ) {
            continue;
        }

        ++feature_count;

        if (
            feature_count >=
            max_features_
        ) {
            break;
        }
    }

    if (feature_count == 0)
        return;

    vocabulary_ =
        Vocabulary{};

    idf_.resize(
        feature_count
    );

    const double N =
        static_cast<double>(
            document_count
        );

    sz output_index = 0;

    for (
        const auto& [term, df] :
        terms_sorted
    ) {

        if (
            df < MinimumDocumentFrequency
        ) {
            continue;
        }

        vocabulary_.add(
            term
        );

        /*
         * Smoothed IDF:
         *
         * log((1 + N)/(1 + df)) + 1
         */

        const double value =
            std::log(
                (1.0 + N) /
                (1.0 +
                 static_cast<double>(df))
            )
            +
            1.0;

        idf_[output_index] =
            (
                std::isfinite(value) &&
                value > 0.0
            )
                ? value
                : 1.0;

        ++output_index;

        if (
            output_index >=
            feature_count
        ) {
            break;
        }
    }

    fitted_ =
        vocabulary_.size() ==
        idf_.size() &&
        vocabulary_.size() > 0;
}


// ============================================================
// Sparse transform
// ============================================================

SparseVector
TfidfVectorizer::transform_sparse(
    sv document
) const {

    SparseVector result;

    if (
        !fitted_ ||
        vocabulary_.size() == 0 ||
        idf_.size() !=
            vocabulary_.size() ||
        document.empty()
    ) {

        return result;
    }

    const auto tokens =
        tokenizer_.tokenize(
            document
        );

    if (tokens.empty())
        return result;

    /*
     * --------------------------------------------------------
     * Instead of map<string,double>:
     *
     *     collect vocabulary IDs
     *     sort integer IDs
     *     aggregate duplicates
     *
     * This is considerably cheaper.
     * --------------------------------------------------------
     */

    vec<u32> ids;

    ids.reserve(
        tokens.size() * 2
    );

    /*
     * Unigrams.
     */

    for (
        const string& token :
        tokens
    ) {

        if (token.empty())
            continue;

        const auto id =
            vocabulary_.find(token);

        if (
            id !=
            Vocabulary::InvalidId
        ) {

            ids.push_back(
                static_cast<u32>(id)
            );
        }
    }

    /*
     * Bigrams.
     */

    if (tokens.size() >= 2) {

        string bigram;

        for (
            sz i = 0;
            i + 1 < tokens.size();
            ++i
        ) {

            if (
                tokens[i].empty() ||
                tokens[i + 1].empty()
            ) {
                continue;
            }

            bigram.clear();

            bigram.reserve(
                tokens[i].size() +
                tokens[i + 1].size() +
                1
            );

            bigram.append(
                tokens[i]
            );

            bigram.push_back('_');

            bigram.append(
                tokens[i + 1]
            );

            const auto id =
                vocabulary_.find(
                    bigram
                );

            if (
                id !=
                Vocabulary::InvalidId
            ) {

                ids.push_back(
                    static_cast<u32>(id)
                );
            }
        }
    }

    if (ids.empty())
        return result;

    sort(
        ids.begin(),
        ids.end()
    );

    result.indices.reserve(
        ids.size()
    );

    result.values.reserve(
        ids.size()
    );

    double squared_norm =
        0.0;

    sz i = 0;

    while (
        i < ids.size()
    ) {

        const u32 index =
            ids[i];

        sz count = 1;

        ++i;

        while (
            i < ids.size() &&
            ids[i] == index
        ) {

            ++count;
            ++i;
        }

        if (
            static_cast<sz>(index) >=
            idf_.size()
        ) {
            continue;
        }

        /*
         * Sublinear TF:
         *
         * 1 + log(tf)
         */

        const double tf =
            1.0 +
            std::log(
                static_cast<double>(count)
            );

        const double value =
            tf *
            idf_[index];

        if (
            !std::isfinite(value) ||
            value <= 0.0
        ) {
            continue;
        }

        result.indices.push_back(
            index
        );

        result.values.push_back(
            value
        );

        squared_norm +=
            value * value;
    }

    if (
        result.empty() ||
        !std::isfinite(squared_norm) ||
        squared_norm <= 0.0
    ) {

        result.clear();

        return result;
    }

    const double inverse_norm =
        1.0 /
        std::sqrt(
            squared_norm
        );

    if (
        !std::isfinite(inverse_norm) ||
        inverse_norm <= 0.0
    ) {

        result.clear();

        return result;
    }

    for (
        double& value :
        result.values
    ) {

        value *=
            inverse_norm;
    }

    return result;
}


// ============================================================
// Dense transform
// ============================================================

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

    const SparseVector sparse =
        transform_sparse(
            document
        );

    const sz count =
        std::min(
            sparse.indices.size(),
            sparse.values.size()
        );

    for (
        sz i = 0;
        i < count;
        ++i
    ) {

        const sz index =
            sparse.indices[i];

        if (
            index < result.size()
        ) {

            result[index] =
                sparse.values[i];
        }
    }

    return result;
}


// ============================================================
// Accessors
// ============================================================

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


sz TfidfVectorizer::vocabulary_size()
    const noexcept {

    return vocabulary_.size();
}


// ============================================================
// Set model data
// ============================================================

bool TfidfVectorizer::set_model_data(
    Vocabulary vocabulary,
    vec<double> idf
) {

    if (
        vocabulary.size() == 0 ||
        vocabulary.size() !=
            idf.size()
    ) {

        fitted_ = false;
        vocabulary_ = Vocabulary{};
        idf_.clear();

        return false;
    }

    for (
        const double value :
        idf
    ) {

        if (
            !std::isfinite(value) ||
            value <= 0.0
        ) {

            fitted_ = false;
            vocabulary_ = Vocabulary{};
            idf_.clear();

            return false;
        }
    }

    vocabulary_ =
        std::move(vocabulary);

    idf_ =
        std::move(idf);

    fitted_ =
        true;

    return true;
}


// ============================================================
// Save
// ============================================================

bool TfidfVectorizer::save(
    const str& path
) const {

    if (
        !fitted_ ||
        vocabulary_.size() == 0 ||
        vocabulary_.size() !=
            idf_.size()
    ) {
        return false;
    }

    ofstream out(
        path,
        ios::binary
    );

    if (!out)
        return false;

    static constexpr char Magic[] =
        "SENTTFIDF4";

    out.write(
        Magic,
        sizeof(Magic)
    );

    const u64 n =
        static_cast<u64>(
            vocabulary_.size()
        );

    out.write(
        reinterpret_cast<const char*>(&n),
        sizeof(n)
    );

    for (
        sz i = 0;
        i < vocabulary_.size();
        ++i
    ) {

        const str& token =
            vocabulary_.token(
                static_cast<
                    Vocabulary::TokenId
                >(i)
            );

        const u64 length =
            static_cast<u64>(
                token.size()
            );

        out.write(
            reinterpret_cast<
                const char*
            >(&length),
            sizeof(length)
        );

        out.write(
            token.data(),
            static_cast<streamsize>(
                token.size()
            )
        );

        out.write(
            reinterpret_cast<
                const char*
            >(&idf_[i]),
            sizeof(double)
        );
    }

    return static_cast<bool>(out);
}


// ============================================================
// Load
// ============================================================

bool TfidfVectorizer::load(
    const str& path
) {

    ifstream in(
        path,
        ios::binary
    );

    if (!in)
        return false;

    static constexpr char Magic[] =
        "SENTTFIDF4";

    char magic[
        sizeof(Magic)
    ]{};

    in.read(
        magic,
        sizeof(magic)
    );

    if (
        !in ||
        string(
            magic,
            sizeof(magic)
        ) !=
        string(
            Magic,
            sizeof(Magic)
        )
    ) {

        return false;
    }

    u64 n = 0;

    in.read(
        reinterpret_cast<char*>(&n),
        sizeof(n)
    );

    if (
        !in ||
        n == 0 ||
        n > 10'000'000ULL
    ) {

        return false;
    }

    Vocabulary vocabulary;

    vec<double> idf(
        static_cast<sz>(n)
    );

    for (
        u64 i = 0;
        i < n;
        ++i
    ) {

        u64 length = 0;

        in.read(
            reinterpret_cast<char*>(&length),
            sizeof(length)
        );

        if (
            !in ||
            length > 1'000'000ULL
        ) {

            return false;
        }

        str token(
            static_cast<sz>(length),
            '\0'
        );

        in.read(
            token.data(),
            static_cast<streamsize>(
                length
            )
        );

        in.read(
            reinterpret_cast<char*>(
                &idf[
                    static_cast<sz>(i)
                ]
            ),
            sizeof(double)
        );

        if (!in)
            return false;

        if (
            !std::isfinite(
                idf[
                    static_cast<sz>(i)
                ]
            ) ||
            idf[
                static_cast<sz>(i)
            ] <= 0.0
        ) {

            return false;
        }

        vocabulary.add(
            token
        );
    }

    return set_model_data(
        std::move(vocabulary),
        std::move(idf)
    );
}

} // namespace sentiment