#include "sentiment/text/tfidf_vectorizer.hpp"

#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

namespace {

constexpr char ModelMagic[] = "SNTFIDF2";
constexpr u32 ModelVersion = 2;


inline void make_bigram(
    const str& left,
    const str& right,
    str& output
)
{
    output.clear();

    output.reserve(
        left.size() +
        right.size() +
        1
    );

    output.append(left);

    output.push_back('_');

    output.append(right);
}

} // namespace


// ============================================================
// Constructor
// ============================================================

TfidfVectorizer::TfidfVectorizer(
    sz max_features,
    sz min_df,
    sz ngram_min,
    sz ngram_max,
    bool sublinear_tf
)
    : max_features_(max_features),
      min_df_(max<sz>(1, min_df)),
      ngram_min_(ngram_min),
      ngram_max_(ngram_max),
      sublinear_tf_(sublinear_tf)
{
    if (
        max_features_ == 0
    ) {

        throw invalid_argument(
            "max_features must be greater than zero."
        );
    }

    if (
        ngram_min_ < 1 ||
        ngram_max_ < ngram_min_
    ) {

        throw invalid_argument(
            "Invalid ngram range."
        );
    }

    /*
     * This optimized implementation intentionally supports:
     *
     *     unigram + bigram
     *
     * which is exactly what the sentiment model requires.
     */
    if (
        ngram_max_ > 2
    ) {

        throw invalid_argument(
            "Only unigram + bigram "
            "(ngram_max <= 2) is supported."
        );
    }
}


// ============================================================
// Fit
// ============================================================

void
TfidfVectorizer::fit(
    const vec<str>& documents
)
{
    unordered_map<str, u32>
        document_frequency;

    document_frequency.max_load_factor(
        0.70f
    );

    document_frequency.reserve(
        max<sz>(
            1024,
            max_features_ * 8
        )
    );


    vec<str> tokens;

    vec<str> unique_terms;

    str bigram;


    u64 document_count = 0;


    // ========================================================
    // Document-frequency pass
    // ========================================================

    for (
        const str& document :
        documents
    ) {

        ++document_count;

        tokens =
            tokenizer_.tokenize(
                document
            );

        if (
            tokens.empty()
        ) {

            continue;
        }


        unique_terms.clear();

        unique_terms.reserve(
            tokens.size() * 2
        );


        // ----------------------------------------------------
        // Unigrams
        // ----------------------------------------------------

        if (
            ngram_min_ <= 1
        ) {

            for (
                const str& token :
                tokens
            ) {

                unique_terms.push_back(
                    token
                );
            }
        }


        // ----------------------------------------------------
        // Bigrams
        // ----------------------------------------------------

        if (
            ngram_min_ <= 2 &&
            ngram_max_ >= 2 &&
            tokens.size() > 1
        ) {

            for (
                sz i = 1;
                i < tokens.size();
                ++i
            ) {

                make_bigram(
                    tokens[i - 1],
                    tokens[i],
                    bigram
                );

                unique_terms.push_back(
                    bigram
                );
            }
        }


        /*
         * A term must count only once per document.
         */
        sort(
            unique_terms.begin(),
            unique_terms.end()
        );

        unique_terms.erase(
            unique(
                unique_terms.begin(),
                unique_terms.end()
            ),
            unique_terms.end()
        );


        // ----------------------------------------------------
        // Update document frequency
        // ----------------------------------------------------

        for (
            const str& term :
            unique_terms
        ) {

            auto [
                iterator,
                inserted
            ] =
                document_frequency.try_emplace(
                    term,
                    0
                );

            (void)inserted;

            if (
                iterator->second <
                numeric_limits<u32>::max()
            ) {

                ++iterator->second;
            }
        }
    }


    // ========================================================
    // Filter by minimum DF
    // ========================================================

    vector<pair<str, u32>>
        candidates;

    candidates.reserve(
        document_frequency.size()
    );


    for (
        auto& entry :
        document_frequency
    ) {

        if (
            entry.second >=
            min_df_
        ) {

            candidates.emplace_back(
                move(entry.first),
                entry.second
            );
        }
    }


    // ========================================================
    // Deterministic feature selection
    // ========================================================

    sort(
        candidates.begin(),
        candidates.end(),
        [](
            const auto& left,
            const auto& right
        ) {

            if (
                left.second !=
                right.second
            ) {

                return
                    left.second >
                    right.second;
            }

            return
                left.first <
                right.first;
        }
    );


    // ========================================================
    // Maximum vocabulary size
    // ========================================================

    if (
        candidates.size() >
        max_features_
    ) {

        candidates.resize(
            max_features_
        );
    }


    // ========================================================
    // Build vocabulary + IDF
    // ========================================================

    Vocabulary new_vocabulary;

    vec<double> new_idf;

    new_idf.reserve(
        candidates.size()
    );


    const double N =
        static_cast<double>(
            document_count
        );


    for (
        const auto& [
            term,
            df
        ] :
        candidates
    ) {

        new_vocabulary.add(
            term
        );


        /*
         * Smooth IDF:
         *
         *     log((N + 1) / (DF + 1)) + 1
         *
         * This is stable for rare and common terms.
         */
        const double idf =
            log(
                (N + 1.0) /
                (
                    static_cast<double>(df) +
                    1.0
                )
            ) + 1.0;


        new_idf.push_back(
            idf
        );
    }


    vocabulary_ =
        move(new_vocabulary);

    idf_ =
        move(new_idf);


    fitted_ =
        !vocabulary_.empty() &&
        vocabulary_.size() ==
        idf_.size();
}


// ============================================================
// Generate feature IDs
// ============================================================

void
TfidfVectorizer::collect_term_ids(
    sv document,
    vec<u32>& ids
) const
{
    ids.clear();


    const vec<str> tokens =
        tokenizer_.tokenize(
            document
        );


    if (
        tokens.empty() ||
        vocabulary_.empty()
    ) {

        return;
    }


    ids.reserve(
        tokens.size() * 2
    );


    // ========================================================
    // Unigrams
    // ========================================================

    if (
        ngram_min_ <= 1
    ) {

        for (
            const str& token :
            tokens
        ) {

            const u32 id =
                vocabulary_.find(
                    token
                );


            if (
                id !=
                Vocabulary::InvalidId
            ) {

                ids.push_back(
                    id
                );
            }
        }
    }


    // ========================================================
    // Bigrams
    // ========================================================

    if (
        ngram_min_ <= 2 &&
        ngram_max_ >= 2 &&
        tokens.size() > 1
    ) {

        str bigram;

        for (
            sz i = 1;
            i < tokens.size();
            ++i
        ) {

            make_bigram(
                tokens[i - 1],
                tokens[i],
                bigram
            );


            const u32 id =
                vocabulary_.find(
                    bigram
                );


            if (
                id !=
                Vocabulary::InvalidId
            ) {

                ids.push_back(
                    id
                );
            }
        }
    }
}


// ============================================================
// Sparse TF-IDF transformation
// ============================================================

SparseVector
TfidfVectorizer::transform_sparse(
    sv document
) const
{
    SparseVector result;
    transform_sparse(document, result);
    return result;
}


void
TfidfVectorizer::transform_sparse(
    sv document,
    SparseVector& result
) const
{
    result.clear();

    if (
        !fitted_ ||
        vocabulary_.empty() ||
        vocabulary_.size() != idf_.size()
    ) {
        return;
    }

    /*
     * Keep the temporary term-id buffer local. The important
     * optimization here is that the caller-owned SparseVector
     * is reused by the training loop, avoiding two allocations
     * per document per epoch.
     */
    vec<u32> ids;
    const vec<str> tokens = tokenizer_.tokenize(document);

    if (tokens.empty()) {
        return;
    }

    ids.reserve(tokens.size() * 2);

    if (ngram_min_ <= 1) {
        for (const str& token : tokens) {
            const u32 id = vocabulary_.find(token);
            if (id != Vocabulary::InvalidId) {
                ids.push_back(id);
            }
        }
    }

    if (
        ngram_min_ <= 2 &&
        ngram_max_ >= 2 &&
        tokens.size() > 1
    ) {
        str bigram;
        bigram.reserve(48);

        for (sz i = 1; i < tokens.size(); ++i) {
            make_bigram(tokens[i - 1], tokens[i], bigram);

            const u32 id = vocabulary_.find(bigram);
            if (id != Vocabulary::InvalidId) {
                ids.push_back(id);
            }
        }
    }

    if (ids.empty()) {
        return;
    }

    sort(ids.begin(), ids.end());

    result.reserve(ids.size());

    for (sz i = 0; i < ids.size();) {
        const u32 feature_id = ids[i];

        sz j = i + 1;
        while (j < ids.size() && ids[j] == feature_id) {
            ++j;
        }

        const double count = static_cast<double>(j - i);
        const double tf = sublinear_tf_ ? 1.0 + log(count) : count;
        const double value = tf * idf_[feature_id];

        if (isfinite(value) && value != 0.0) {
            result.indices.push_back(feature_id);
            result.values.push_back(value);
        }

        i = j;
    }

    double norm_squared = 0.0;
    for (double value : result.values) {
        norm_squared += value * value;
    }

    if (norm_squared > 0.0 && isfinite(norm_squared)) {
        const double inverse_norm = 1.0 / sqrt(norm_squared);
        for (double& value : result.values) {
            value *= inverse_norm;
        }
    }
}

// ============================================================
// Dense transformation
// ============================================================

vec<double>
TfidfVectorizer::transform(
    sv document
) const
{
    vec<double> result(
        vocabulary_.size(),
        0.0
    );


    const SparseVector sparse =
        transform_sparse(
            document
        );


    for (
        sz i = 0;
        i < sparse.indices.size();
        ++i
    ) {

        result[
            sparse.indices[i]
        ] =
            sparse.values[i];
    }


    return result;
}


// ============================================================
// Accessors
// ============================================================

sz
TfidfVectorizer::vocabulary_size()
    const noexcept
{
    return vocabulary_.size();
}


const Vocabulary&
TfidfVectorizer::vocabulary()
    const noexcept
{
    return vocabulary_;
}


const vec<double>&
TfidfVectorizer::idf()
    const noexcept
{
    return idf_;
}


// ============================================================
// Model data
// ============================================================

bool
TfidfVectorizer::set_model_data(
    Vocabulary vocabulary,
    vec<double> idf
)
{
    if (
        vocabulary.empty() ||
        vocabulary.size() !=
        idf.size()
    ) {

        fitted_ = false;

        vocabulary_ = {};

        idf_.clear();

        return false;
    }


    for (
        const double value :
        idf
    ) {

        if (
            !isfinite(value) ||
            value <= 0.0
        ) {

            fitted_ = false;

            return false;
        }
    }


    vocabulary_ =
        move(vocabulary);

    idf_ =
        move(idf);

    fitted_ =
        true;


    return true;
}


// ============================================================
// Save model
// ============================================================

bool
TfidfVectorizer::save(
    const str& path
) const
{
    if (
        !fitted_ ||
        vocabulary_.size() !=
        idf_.size()
    ) {

        return false;
    }


    ofstream output(
        path,
        ios::binary |
        ios::trunc
    );


    if (
        !output
    ) {

        return false;
    }


    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    output.write(
        ModelMagic,
        sizeof(ModelMagic) - 1
    );


    const u32 version =
        ModelVersion;


    const u64 feature_count =
        static_cast<u64>(
            vocabulary_.size()
        );


    const u64 min_df =
        static_cast<u64>(
            min_df_
        );


    const u64 ngram_min =
        static_cast<u64>(
            ngram_min_
        );


    const u64 ngram_max =
        static_cast<u64>(
            ngram_max_
        );


    const u8 sublinear =
        sublinear_tf_
            ? 1
            : 0;


    output.write(
        reinterpret_cast<const char*>(
            &version
        ),
        sizeof(version)
    );


    output.write(
        reinterpret_cast<const char*>(
            &feature_count
        ),
        sizeof(feature_count)
    );


    output.write(
        reinterpret_cast<const char*>(
            &min_df
        ),
        sizeof(min_df)
    );


    output.write(
        reinterpret_cast<const char*>(
            &ngram_min
        ),
        sizeof(ngram_min)
    );


    output.write(
        reinterpret_cast<const char*>(
            &ngram_max
        ),
        sizeof(ngram_max)
    );


    output.write(
        reinterpret_cast<const char*>(
            &sublinear
        ),
        sizeof(sublinear)
    );


    // --------------------------------------------------------
    // Features
    // --------------------------------------------------------

    for (
        sz i = 0;
        i < vocabulary_.size();
        ++i
    ) {

        const str& term =
            vocabulary_.term(
                static_cast<u32>(i)
            );


        const u64 length =
            static_cast<u64>(
                term.size()
            );


        output.write(
            reinterpret_cast<const char*>(
                &length
            ),
            sizeof(length)
        );


        output.write(
            term.data(),
            static_cast<streamsize>(
                term.size()
            )
        );


        output.write(
            reinterpret_cast<const char*>(
                &idf_[i]
            ),
            sizeof(double)
        );


        if (
            !output
        ) {

            return false;
        }
    }


    return static_cast<bool>(
        output
    );
}


// ============================================================
// Load model
// ============================================================

bool
TfidfVectorizer::load(
    const str& path
)
{
    ifstream input(
        path,
        ios::binary
    );


    if (
        !input
    ) {

        return false;
    }


    // --------------------------------------------------------
    // Magic
    // --------------------------------------------------------

    char magic[
        sizeof(ModelMagic) - 1
    ]{};


    input.read(
        magic,
        sizeof(magic)
    );


    if (
        !input ||
        memcmp(
            magic,
            ModelMagic,
            sizeof(magic)
        ) != 0
    ) {

        return false;
    }


    // --------------------------------------------------------
    // Header
    // --------------------------------------------------------

    u32 version = 0;

    u64 feature_count = 0;

    u64 stored_min_df = 1;

    u64 stored_ngram_min = 1;

    u64 stored_ngram_max = 2;

    u8 stored_sublinear = 1;


    input.read(
        reinterpret_cast<char*>(
            &version
        ),
        sizeof(version)
    );


    input.read(
        reinterpret_cast<char*>(
            &feature_count
        ),
        sizeof(feature_count)
    );


    input.read(
        reinterpret_cast<char*>(
            &stored_min_df
        ),
        sizeof(stored_min_df)
    );


    input.read(
        reinterpret_cast<char*>(
            &stored_ngram_min
        ),
        sizeof(stored_ngram_min)
    );


    input.read(
        reinterpret_cast<char*>(
            &stored_ngram_max
        ),
        sizeof(stored_ngram_max)
    );


    input.read(
        reinterpret_cast<char*>(
            &stored_sublinear
        ),
        sizeof(stored_sublinear)
    );


    if (
        !input ||
        version != ModelVersion ||
        feature_count == 0 ||
        feature_count >
            static_cast<u64>(
                numeric_limits<u32>::max()
            )
    ) {

        return false;
    }


    if (
        stored_ngram_min < 1 ||
        stored_ngram_max <
            stored_ngram_min ||
        stored_ngram_max > 2 ||
        stored_sublinear > 1
    ) {

        return false;
    }


    // --------------------------------------------------------
    // Load vocabulary
    // --------------------------------------------------------

    Vocabulary new_vocabulary;

    vec<double> new_idf;

    new_idf.reserve(
        static_cast<sz>(
            feature_count
        )
    );


    for (
        u64 i = 0;
        i < feature_count;
        ++i
    ) {

        u64 length = 0;


        input.read(
            reinterpret_cast<char*>(
                &length
            ),
            sizeof(length)
        );


        if (
            !input ||
            length > 1'000'000ULL
        ) {

            return false;
        }


        str term(
            static_cast<sz>(
                length
            ),
            '\0'
        );


        input.read(
            term.data(),
            static_cast<streamsize>(
                length
            )
        );


        double idf = 0.0;


        input.read(
            reinterpret_cast<char*>(
                &idf
            ),
            sizeof(idf)
        );


        if (
            !input ||
            term.empty() ||
            !isfinite(idf) ||
            idf <= 0.0
        ) {

            return false;
        }


        new_vocabulary.add(
            term
        );


        new_idf.push_back(
            idf
        );
    }


    if (
        new_vocabulary.size() !=
        static_cast<sz>(
            feature_count
        )
    ) {

        return false;
    }


    // --------------------------------------------------------
    // Commit only after successful load
    // --------------------------------------------------------

    vocabulary_ =
        move(new_vocabulary);

    idf_ =
        move(new_idf);


    min_df_ =
        static_cast<sz>(
            stored_min_df
        );


    ngram_min_ =
        static_cast<sz>(
            stored_ngram_min
        );


    ngram_max_ =
        static_cast<sz>(
            stored_ngram_max
        );


    sublinear_tf_ =
        stored_sublinear != 0;


    max_features_ =
        vocabulary_.size();


    fitted_ =
        true;


    return true;
}

} // namespace sentiment