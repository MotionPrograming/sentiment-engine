#include "sentiment/io/dataset_reader.hpp"
#include "sentiment/ml/evaluator.hpp"
#include "sentiment/ml/linear_svm.hpp"
#include "sentiment/text/tfidf_vectorizer.hpp"

#include <bits/stdc++.h>

using namespace std;

namespace
{

    using sentiment::EvaluationResult;
    using sentiment::Evaluator;
    using sentiment::LinearSVM;
    using sentiment::ParquetDatasetReader;
    using sentiment::Review;
    using sentiment::Sentiment;
    using sentiment::SparseVector;
    using sentiment::TfidfVectorizer;
    using sentiment::Tokenizer;
    using sentiment::Vocabulary;

    using sz = size_t;
    using u8 = uint8_t;
    using u32 = uint32_t;
    using u64 = uint64_t;

    constexpr sz ClassCount = LinearSVM::ClassCount;

    constexpr sz MaxFeatures = 250000;
    constexpr sz MinimumDocumentFrequency = 3;
    constexpr sz ParquetBatchSize = 65'536;

    constexpr sz Epochs = 50;

    constexpr double InitialLearningRate = 0.0010;
    constexpr double MinimumLearningRate = 0.00008;
    constexpr double LearningRateDecay = 0.12;

    constexpr double Regularization = 0.00005;

    constexpr sz ValidationInterval = 2;
    constexpr sz EarlyStoppingPatience = 3;

    constexpr double MinimumMacroF1Improvement = 0.0003;

    /*
     * ============================================================
     * CLASS WEIGHTS
     * ============================================================
     *
     * Negative = 1.0
     * Neutral  = 4.5
     * Positive = 1.0
     */
    constexpr double NegativeWeightMultiplier = 1.00;
    constexpr double NeutralWeightMultiplier = 2.40;
    constexpr double PositiveWeightMultiplier = 1.00;

    constexpr u64 SplitSeed =
        0x9E3779B97F4A7C15ULL;

    /*
     * Increase this whenever cache format or split logic changes.
     */
    constexpr u64 CacheVersion = 3;

    constexpr array<double, ClassCount>
        ManualClassWeightMultipliers{
            NegativeWeightMultiplier,
            NeutralWeightMultiplier,
            PositiveWeightMultiplier};

    struct DatasetStatistics
    {

        array<u64, ClassCount> class_counts{};

        u64 total_rows = 0;

        u64 duplicate_rows = 0;

        u64 conflicting_duplicates = 0;
    };

    struct CachePaths
    {

        filesystem::path train;

        filesystem::path validation;

        filesystem::path test;

        filesystem::path manifest;
    };

    struct CacheHeader
    {

        char magic[8];

        u64 version;

        u64 row_count;
    };

    struct CacheManifest
    {

        u64 version;

        u64 dataset_size;

        u64 dataset_timestamp;

        u64 split_seed;

        u64 train_rows;

        u64 validation_rows;

        u64 test_rows;
    };

    struct CacheData
    {

        vector<Review> rows;
    };

    /*
     * ============================================================
     * HASHING
     * ============================================================
     */

    u64 hash_bytes(string_view text)
    {
        /*
         * FNV-1a 64-bit
         */
        u64 hash =
            14695981039346656037ULL;

        for (unsigned char c : text)
        {

            hash ^= static_cast<u64>(c);

            hash *= 1099511628211ULL;
        }

        return hash;
    }

    string normalize_for_hash(string_view text)
    {
        string normalized;

        normalized.reserve(
            text.size());

        for (unsigned char c : text)
        {

            if (isalnum(c))
            {

                normalized.push_back(
                    static_cast<char>(
                        tolower(c)));
            }
            else if (isspace(c))
            {

                if (
                    !normalized.empty() &&
                    normalized.back() != ' ')
                {

                    normalized.push_back(' ');
                }
            }
        }

        while (
            !normalized.empty() &&
            normalized.back() == ' ')
        {

            normalized.pop_back();
        }

        return normalized;
    }

    u64 hash_normalized(string_view text)
    {
        const string normalized =
            normalize_for_hash(text);

        return hash_bytes(normalized);
    }

    /*
     * ============================================================
     * SENTIMENT CONVERSION
     * ============================================================
     */

    int sentiment_to_int(
        Sentiment sentiment)
    {
        return static_cast<int>(
            sentiment);
    }

    Sentiment int_to_sentiment(
        int value)
    {
        if (
            value < 0 ||
            value >= static_cast<int>(ClassCount))
        {

            throw runtime_error(
                "Invalid sentiment label in cache.");
        }

        return static_cast<Sentiment>(
            value);
    }

    /*
     * ============================================================
     * CACHE PATHS
     * ============================================================
     */

    CachePaths make_cache_paths(
        const filesystem::path &cache_directory)
    {
        return {
            cache_directory / "train.bin",
            cache_directory / "validation.bin",
            cache_directory / "test.bin",
            cache_directory / "manifest.bin"};
    }

    /*
     * ============================================================
     * DATASET TIMESTAMP
     * ============================================================
     */

    u64 dataset_timestamp(
        const filesystem::path &dataset_path)
    {
        const auto time =
            filesystem::last_write_time(
                dataset_path);

        return static_cast<u64>(
            time.time_since_epoch().count());
    }

    /*
     * ============================================================
     * CACHE WRITE
     * ============================================================
     */

    bool write_cache(
        const filesystem::path &path,
        const vector<Review> &rows)
    {
        ofstream output(
            path,
            ios::binary | ios::trunc);

        if (!output)
        {
            return false;
        }

        CacheHeader header{};

        memcpy(
            header.magic,
            "SNTCACHE",
            8);

        header.version =
            CacheVersion;

        header.row_count =
            rows.size();

        output.write(
            reinterpret_cast<const char *>(&header),
            sizeof(header));

        for (
            const Review &review :
            rows)
        {

            const u8 label =
                static_cast<u8>(
                    sentiment_to_int(
                        review.sentiment));

            const u64 text_size =
                static_cast<u64>(
                    review.text.size());

            output.write(
                reinterpret_cast<const char *>(&label),
                sizeof(label));

            output.write(
                reinterpret_cast<const char *>(&text_size),
                sizeof(text_size));

            output.write(
                review.text.data(),
                static_cast<streamsize>(
                    text_size));

            if (!output)
            {
                return false;
            }
        }

        return true;
    }

    /*
     * ============================================================
     * CACHE READ
     * ============================================================
     */

    CacheData read_cache(
        const filesystem::path &path)
    {
        ifstream input(
            path,
            ios::binary);

        if (!input)
        {

            throw runtime_error(
                "Cannot open cache: " +
                path.string());
        }

        CacheHeader header{};

        input.read(
            reinterpret_cast<char *>(&header),
            sizeof(header));

        if (!input)
        {

            throw runtime_error(
                "Invalid cache header: " +
                path.string());
        }

        if (
            memcmp(
                header.magic,
                "SNTCACHE",
                8) != 0)
        {

            throw runtime_error(
                "Invalid cache magic: " +
                path.string());
        }

        if (
            header.version !=
            CacheVersion)
        {

            throw runtime_error(
                "Cache version mismatch: " +
                path.string());
        }

        CacheData data;

        data.rows.reserve(
            static_cast<sz>(
                header.row_count));

        for (
            u64 i = 0;
            i < header.row_count;
            ++i)
        {

            u8 label = 0;

            u64 text_size = 0;

            input.read(
                reinterpret_cast<char *>(&label),
                sizeof(label));

            input.read(
                reinterpret_cast<char *>(&text_size),
                sizeof(text_size));

            if (!input)
            {

                throw runtime_error(
                    "Corrupted cache: " +
                    path.string());
            }

            /*
             * Basic corruption protection.
             */
            if (
                text_size > 10'000'000ULL)
            {

                throw runtime_error(
                    "Invalid text size in cache: " +
                    path.string());
            }

            string text;

            text.resize(
                static_cast<sz>(
                    text_size));

            input.read(
                text.data(),
                static_cast<streamsize>(
                    text_size));

            if (!input)
            {

                throw runtime_error(
                    "Corrupted cache text: " +
                    path.string());
            }

            Review review;

            review.sentiment =
                int_to_sentiment(
                    static_cast<int>(
                        label));

            review.text =
                move(text);

            data.rows.push_back(
                move(review));
        }

        return data;
    }

    /*
     * ============================================================
     * MANIFEST
     * ============================================================
     */

    bool write_manifest(
        const filesystem::path &path,
        const CacheManifest &manifest)
    {
        ofstream output(
            path,
            ios::binary | ios::trunc);

        if (!output)
        {
            return false;
        }

        output.write(
            reinterpret_cast<const char *>(&manifest),
            sizeof(manifest));

        return static_cast<bool>(
            output);
    }

    bool read_manifest(
        const filesystem::path &path,
        CacheManifest &manifest)
    {
        ifstream input(
            path,
            ios::binary);

        if (!input)
        {
            return false;
        }

        input.read(
            reinterpret_cast<char *>(&manifest),
            sizeof(manifest));

        return static_cast<bool>(
            input);
    }

    /*
     * ============================================================
     * CACHE VALIDATION
     * ============================================================
     */

    bool cache_is_valid(
        const filesystem::path &dataset_path,
        const CachePaths &paths)
    {
        if (
            !filesystem::exists(paths.train) ||
            !filesystem::exists(paths.validation) ||
            !filesystem::exists(paths.test) ||
            !filesystem::exists(paths.manifest))
        {

            return false;
        }

        CacheManifest manifest{};

        if (
            !read_manifest(
                paths.manifest,
                manifest))
        {

            return false;
        }

        if (
            manifest.version !=
            CacheVersion)
        {

            return false;
        }

        const u64 current_size =
            filesystem::file_size(
                dataset_path);

        const u64 current_timestamp =
            dataset_timestamp(
                dataset_path);

        if (
            manifest.dataset_size !=
            current_size)
        {

            return false;
        }

        if (
            manifest.dataset_timestamp !=
            current_timestamp)
        {

            return false;
        }

        if (
            manifest.split_seed !=
            SplitSeed)
        {

            return false;
        }

        return true;
    }

    /*
     * ============================================================
     * BUILD DISK CACHE
     * ============================================================
     */

    DatasetStatistics build_cache(
        const filesystem::path &dataset_path,
        const CachePaths &paths)
    {
        cout << "\n";
        cout << "========================================\n";
        cout << " Building disk cache\n";
        cout << "========================================\n";

        filesystem::create_directories(
            paths.train.parent_path());

        ParquetDatasetReader reader(
            dataset_path.string(),
            ParquetBatchSize);

        if (!reader.good())
        {

            throw runtime_error(
                "Failed to open parquet dataset: " +
                dataset_path.string());
        }

        vector<Review> train_rows;

        vector<Review> validation_rows;

        vector<Review> test_rows;

        train_rows.reserve(
            2'500'000);

        validation_rows.reserve(
            350'000);

        test_rows.reserve(
            350'000);

        DatasetStatistics statistics;

        unordered_map<u64, u8> seen_hashes;

        seen_hashes.reserve(
            4'000'000);

        Review review;

        /*
         * IMPORTANT:
         *
         * ParquetDatasetReader does NOT have for_each().
         * Use next(review).
         */
        while (
            reader.next(review))
        {

            if (review.text.empty())
            {
                continue;
            }

            const string normalized =
                normalize_for_hash(
                    review.text);

            if (normalized.empty())
            {
                continue;
            }

            const u64 text_hash =
                hash_bytes(
                    normalized);

            const u8 label =
                static_cast<u8>(
                    sentiment_to_int(
                        review.sentiment));

            const auto [it, inserted] =
                seen_hashes.emplace(
                    text_hash,
                    label);

            if (!inserted)
            {

                ++statistics.duplicate_rows;

                if (
                    it->second !=
                    label)
                {

                    ++statistics.conflicting_duplicates;
                }

                continue;
            }

            ++statistics.total_rows;

            if (
                label < ClassCount)
            {

                ++statistics.class_counts[label];
            }

            /*
             * Deterministic 80 / 10 / 10 split.
             */
            const u64 bucket =
                (text_hash ^ SplitSeed) %
                1000ULL;

            if (
                bucket < 800ULL)
            {

                train_rows.push_back(
                    review);
            }
            else if (
                bucket < 900ULL)
            {

                validation_rows.push_back(
                    review);
            }
            else
            {

                test_rows.push_back(
                    review);
            }
        }

        cout << "\n";

        cout
            << "Rows read          : "
            << reader.rows_read()
            << '\n';

        cout
            << "Unique rows        : "
            << statistics.total_rows
            << '\n';

        cout
            << "Duplicate rows     : "
            << statistics.duplicate_rows
            << '\n';

        cout
            << "Conflict duplicates: "
            << statistics.conflicting_duplicates
            << '\n';

        cout << "\nClass distribution:\n";

        cout
            << "Negative           : "
            << statistics.class_counts[0]
            << '\n';

        cout
            << "Neutral            : "
            << statistics.class_counts[1]
            << '\n';

        cout
            << "Positive           : "
            << statistics.class_counts[2]
            << '\n';

        cout << "\nSplit:\n";

        cout
            << "Train              : "
            << train_rows.size()
            << '\n';

        cout
            << "Validation         : "
            << validation_rows.size()
            << '\n';

        cout
            << "Test               : "
            << test_rows.size()
            << '\n';

        cout << "\nWriting cache...\n";

        if (
            !write_cache(
                paths.train,
                train_rows))
        {

            throw runtime_error(
                "Failed to write train cache.");
        }

        if (
            !write_cache(
                paths.validation,
                validation_rows))
        {

            throw runtime_error(
                "Failed to write validation cache.");
        }

        if (
            !write_cache(
                paths.test,
                test_rows))
        {

            throw runtime_error(
                "Failed to write test cache.");
        }

        CacheManifest manifest{};

        manifest.version =
            CacheVersion;

        manifest.dataset_size =
            filesystem::file_size(
                dataset_path);

        manifest.dataset_timestamp =
            dataset_timestamp(
                dataset_path);

        manifest.split_seed =
            SplitSeed;

        manifest.train_rows =
            train_rows.size();

        manifest.validation_rows =
            validation_rows.size();

        manifest.test_rows =
            test_rows.size();

        if (
            !write_manifest(
                paths.manifest,
                manifest))
        {

            throw runtime_error(
                "Failed to write cache manifest.");
        }

        cout
            << "Cache build complete.\n";

        return statistics;
    }

    /*
     * ============================================================
     * CLASS WEIGHTS
     * ============================================================
     */

    array<double, ClassCount>
    make_class_weights(
        const DatasetStatistics &statistics)
    {
        /*
         * Manual class weighting.
         *
         * Statistics are intentionally accepted so this can later
         * be replaced by automatic weighting without changing the
         * training pipeline.
         */
        (void)statistics;

        return ManualClassWeightMultipliers;
    }

    /*
     * ============================================================
     * STREAMING TF-IDF VOCABULARY BUILDER
     * ============================================================
     */

    class StreamingTfidfBuilder
    {

    public:
        StreamingTfidfBuilder(
            const Tokenizer &tokenizer,
            sz max_features,
            sz minimum_document_frequency)
            : tokenizer_(tokenizer),
              max_features_(max_features),
              minimum_document_frequency_(
                  minimum_document_frequency)
        {
        }

        Vocabulary build(
            const vector<Review> &rows,
            vector<double> &idf)
        {
            cout << "\n";

            cout
                << "========================================\n";

            cout
                << " Building TF-IDF vocabulary\n";

            cout
                << "========================================\n";

            unordered_map<string, u32>
                document_frequency;

            document_frequency.reserve(
                max_features_ * 4);

            u64 document_count = 0;

            for (
                const Review &review :
                rows)
            {

                ++document_count;

                /*
                 * Actual Tokenizer API:
                 *
                 * tokenize(document)
                 *
                 * returns the token vector.
                 */
                const vector<string> tokens =
                    tokenizer_.tokenize(
                        review.text);

                if (
                    tokens.empty())
                {

                    continue;
                }

                /*
                 * Unique terms within this document.
                 *
                 * Both unigram and bigram features.
                 */
                unordered_set<string>
                    unique_terms;

                unique_terms.reserve(
                    tokens.size() * 2);

                /*
                 * Unigrams.
                 */
                for (
                    const string &token :
                    tokens)
                {

                    unique_terms.insert(
                        token);
                }

                /*
                 * Bigrams.
                 */
                for (
                    sz i = 1;
                    i < tokens.size();
                    ++i)
                {

                    string bigram;

                    bigram.reserve(
                        tokens[i - 1].size() +
                        tokens[i].size() +
                        1);

                    bigram +=
                        tokens[i - 1];

                    bigram.push_back(
                        '_');

                    bigram +=
                        tokens[i];

                    unique_terms.insert(
                        move(bigram));
                }

                /*
                 * Update document frequency.
                 */
                for (
                    const string &term :
                    unique_terms)
                {

                    auto [it, inserted] =
                        document_frequency.try_emplace(
                            term,
                            0);

                    (void)inserted;

                    if (
                        it->second <
                        numeric_limits<u32>::max())
                    {

                        ++it->second;
                    }
                }
            }

            cout
                << "Documents          : "
                << document_count
                << '\n';

            cout
                << "Unique terms       : "
                << document_frequency.size()
                << '\n';

            /*
             * Filter by minimum document frequency.
             */
            vector<pair<string, u32>>
                candidates;

            candidates.reserve(
                document_frequency.size());

            for (
                const auto &[term, df] :
                document_frequency)
            {

                if (
                    df >=
                    minimum_document_frequency_)
                {

                    candidates.emplace_back(
                        term,
                        df);
                }
            }

            /*
             * Highest document frequency first.
             * Lexical order gives deterministic ties.
             */
            sort(
                candidates.begin(),
                candidates.end(),
                [](const auto &a, const auto &b)
                {
                    if (
                        a.second !=
                        b.second)
                    {

                        return a.second >
                               b.second;
                    }

                    return a.first <
                           b.first;
                });

            /*
             * Limit vocabulary size.
             */
            if (
                candidates.size() >
                max_features_)
            {

                candidates.resize(
                    max_features_);
            }

            /*
             * Build vocabulary using the actual API:
             *
             * Vocabulary::add(token)
             */
            Vocabulary vocabulary;

            idf.clear();

            idf.reserve(
                candidates.size());

            const double N =
                static_cast<double>(
                    document_count);

            for (
                const auto &[term, df] :
                candidates)
            {

                vocabulary.add(
                    term);

                /*
                 * Smooth IDF:
                 *
                 * log((1 + N) / (1 + df)) + 1
                 */
                const double value =
                    log(
                        (1.0 + N) /
                        (1.0 +
                         static_cast<double>(
                             df))) +
                    1.0;

                idf.push_back(
                    value);
            }

            cout
                << "Selected features  : "
                << vocabulary.size()
                << '\n';

            cout
                << "Minimum DF         : "
                << minimum_document_frequency_
                << '\n';

            cout
                << "Maximum features   : "
                << max_features_
                << '\n';

            return vocabulary;
        }

    private:
        const Tokenizer &tokenizer_;

        sz max_features_;

        sz minimum_document_frequency_;
    };

    /*
     * ============================================================
     * MODEL EVALUATION
     * ============================================================
     */

    EvaluationResult evaluate_model(
        const LinearSVM &svm,
        TfidfVectorizer &vectorizer,
        const vector<Review> &rows)
    {
        vector<Sentiment> actual;

        vector<Sentiment> predicted;

        actual.reserve(
            rows.size());

        predicted.reserve(
            rows.size());

        SparseVector features;
        features.indices.reserve(128);
        features.values.reserve(128);

        for (const Review &review : rows)
        {

            vectorizer.transform_sparse(
                review.text,
                features);

            const Sentiment prediction =
                svm.predict(
                    features);

            actual.push_back(
                review.sentiment);

            predicted.push_back(
                prediction);
        }

        /*
         * Actual Evaluator API:
         *
         * Evaluator::evaluate(
         *     actual,
         *     predicted
         * )
         */
        return Evaluator::evaluate(
            actual,
            predicted);
    }

    /*
     * ============================================================
     * PRINT EVALUATION
     * ============================================================
     */

    void print_evaluation(
        const string &name,
        const EvaluationResult &result)
    {
        cout << "\n";

        cout
            << "----------------------------------------\n";

        cout
            << name
            << '\n';

        cout
            << "----------------------------------------\n";

        cout
            << fixed
            << setprecision(4);

        cout
            << "Accuracy    : "
            << result.accuracy
            << '\n';

        cout
            << "Macro F1    : "
            << result.macro_f1
            << '\n';

        cout
            << "Weighted F1 : "
            << result.weighted_f1
            << '\n';

        for (
            sz c = 0;
            c < ClassCount;
            ++c)
        {

            /*
             * Support =
             * sum of confusion row.
             */
            u64 support = 0;

            for (
                sz predicted_class = 0;
                predicted_class < ClassCount;
                ++predicted_class)
            {

                support +=
                    result.confusion[c]
                                    [predicted_class];
            }

            cout
                << "\nClass "
                << c
                << '\n';

            cout
                << "  Precision: "
                << result.precision[c]
                << '\n';

            cout
                << "  Recall   : "
                << result.recall[c]
                << '\n';

            cout
                << "  F1       : "
                << result.f1[c]
                << '\n';

            cout
                << "  Support  : "
                << support
                << '\n';
        }

        cout
            << "\nConfusion Matrix:\n";

        cout
            << "                 Predicted\n";

        cout
            << "              Neg       Neu       Pos\n";

        for (
            sz actual_class = 0;
            actual_class < ClassCount;
            ++actual_class)
        {

            cout
                << "Actual ";

            if (
                actual_class == 0)
            {

                cout
                    << "Neg    ";
            }
            else if (
                actual_class == 1)
            {

                cout
                    << "Neu    ";
            }
            else
            {

                cout
                    << "Pos    ";
            }

            for (
                sz predicted_class = 0;
                predicted_class < ClassCount;
                ++predicted_class)
            {

                cout
                    << setw(10)
                    << result.confusion
                           [actual_class]
                           [predicted_class];
            }

            cout
                << '\n';
        }
    }

    /*
     * ============================================================
     * LEARNING RATE SCHEDULE
     * ============================================================
     */

    double learning_rate_for_epoch(
        sz epoch)
    {
        const double decay =
            1.0 +
            LearningRateDecay *
                static_cast<double>(
                    epoch - 1);

        const double lr =
            InitialLearningRate /
            decay;

        return max(
            MinimumLearningRate,
            lr);
    }

} // namespace

/*
 * ============================================================
 * MAIN
 * ============================================================
 */

int main(
    int argc,
    char **argv)
{
    try
    {

        /*
         * --------------------------------------------------------
         * Command-line arguments
         * --------------------------------------------------------
         *
         * argv[1] = dataset
         * argv[2] = model directory
         * argv[3] = cache directory
         */
        if (
            argc < 4)
        {

            cerr
                << "Usage:\n"
                << "  "
                << argv[0]
                << " <dataset.parquet>"
                << " <model_dir>"
                << " <cache_dir>\n";

            return EXIT_FAILURE;
        }

        const filesystem::path dataset_path =
            argv[1];

        const filesystem::path model_directory =
            argv[2];

        const filesystem::path cache_directory =
            argv[3];

        cout
            << "========================================\n";

        cout
            << " SentimentEngine\n";

        cout
            << " Production Linear SVM Training\n";

        cout
            << "========================================\n";

        cout
            << "\nConfiguration\n";

        cout
            << "----------------------------------------\n";

        cout
            << "Dataset            : "
            << dataset_path
            << '\n';

        cout
            << "Model directory    : "
            << model_directory
            << '\n';

        cout
            << "Cache directory    : "
            << cache_directory
            << '\n';

        cout
            << "Max features       : "
            << MaxFeatures
            << '\n';

        cout
            << "Minimum DF         : "
            << MinimumDocumentFrequency
            << '\n';

        cout
            << "Epochs             : "
            << Epochs
            << '\n';

        cout
            << "Initial LR         : "
            << InitialLearningRate
            << '\n';

        cout
            << "Minimum LR         : "
            << MinimumLearningRate
            << '\n';

        cout
            << "LR decay           : "
            << LearningRateDecay
            << '\n';

        cout
            << "Regularization     : "
            << Regularization
            << '\n';

        cout
            << "Validation interval: "
            << ValidationInterval
            << '\n';

        cout
            << "Early patience     : "
            << EarlyStoppingPatience
            << '\n';

        cout
            << "\nClass weights\n";

        cout
            << "----------------------------------------\n";

        cout
            << "Negative           : "
            << NegativeWeightMultiplier
            << '\n';

        cout
            << "Neutral            : "
            << NeutralWeightMultiplier
            << '\n';

        cout
            << "Positive           : "
            << PositiveWeightMultiplier
            << '\n';

        /*
         * --------------------------------------------------------
         * Verify dataset
         * --------------------------------------------------------
         */

        if (
            !filesystem::exists(
                dataset_path))
        {

            throw runtime_error(
                "Dataset does not exist: " +
                dataset_path.string());
        }

        filesystem::create_directories(
            model_directory);

        filesystem::create_directories(
            cache_directory);

        /*
         * --------------------------------------------------------
         * Cache
         * --------------------------------------------------------
         */

        const CachePaths cache_paths =
            make_cache_paths(
                cache_directory);

        DatasetStatistics statistics;

        vector<Review> train_rows;

        vector<Review> validation_rows;

        vector<Review> test_rows;

        if (
            cache_is_valid(
                dataset_path,
                cache_paths))
        {

            cout
                << "\n";

            cout
                << "========================================\n";

            cout
                << " Using existing disk cache\n";

            cout
                << "========================================\n";

            train_rows =
                read_cache(
                    cache_paths.train)
                    .rows;

            validation_rows =
                read_cache(
                    cache_paths.validation)
                    .rows;

            test_rows =
                read_cache(
                    cache_paths.test)
                    .rows;
        }
        else
        {

            cout
                << "\nCache is missing or outdated.\n";

            statistics =
                build_cache(
                    dataset_path,
                    cache_paths);

            train_rows =
                read_cache(
                    cache_paths.train)
                    .rows;

            validation_rows =
                read_cache(
                    cache_paths.validation)
                    .rows;

            test_rows =
                read_cache(
                    cache_paths.test)
                    .rows;
        }

        cout
            << "\nDataset ready.\n";

        cout
            << "Train rows      : "
            << train_rows.size()
            << '\n';

        cout
            << "Validation rows : "
            << validation_rows.size()
            << '\n';

        cout
            << "Test rows       : "
            << test_rows.size()
            << '\n';

        /*
         * --------------------------------------------------------
         * Tokenizer
         * --------------------------------------------------------
         */

        Tokenizer tokenizer;

        /*
         * --------------------------------------------------------
         * TF-IDF
         * --------------------------------------------------------
         *
         * Vocabulary is built ONLY from training data.
         */

        StreamingTfidfBuilder tfidf_builder(
            tokenizer,
            MaxFeatures,
            MinimumDocumentFrequency);

        vector<double> idf;

        Vocabulary vocabulary =
            tfidf_builder.build(
                train_rows,
                idf);

        if (
            vocabulary.size() !=
            idf.size())
        {

            throw runtime_error(
                "TF-IDF vocabulary/IDF size mismatch.");
        }

        TfidfVectorizer vectorizer;

        /*
         * Actual API:
         *
         * set_model_data(
         *     Vocabulary,
         *     vector<double>
         * )
         */
        if (
            !vectorizer.set_model_data(
                move(vocabulary),
                move(idf)))
        {

            throw runtime_error(
                "Failed to initialize TF-IDF model data.");
        }

        /*
         * --------------------------------------------------------
         * Linear SVM
         * --------------------------------------------------------
         */

        LinearSVM svm(
            vectorizer.vocabulary_size());

        /*
         * Manual class weights.
         *
         * Negative = 1.0
         * Neutral  = 4.5
         * Positive = 1.0
         */

        const auto requested_class_weights =
            make_class_weights(
                statistics);

        svm.set_class_weights(
            requested_class_weights);

        /*
         * Print actual weights stored by SVM.
         */

        const auto &actual_weights =
            svm.class_weights();

        cout
            << "\n";

        cout
            << "========================================\n";

        cout
            << " SVM Class Weights\n";

        cout
            << "========================================\n";

        cout
            << fixed
            << setprecision(4);

        cout
            << "Requested Negative : "
            << requested_class_weights[0]
            << '\n';

        cout
            << "Requested Neutral  : "
            << requested_class_weights[1]
            << '\n';

        cout
            << "Requested Positive : "
            << requested_class_weights[2]
            << '\n';

        cout
            << "\nActual SVM weights:\n";

        cout
            << "Negative           : "
            << actual_weights[0]
            << '\n';

        cout
            << "Neutral            : "
            << actual_weights[1]
            << '\n';

        cout
            << "Positive           : "
            << actual_weights[2]
            << '\n';

        /*
         * Verify Neutral weight.
         */

        if (
            fabs(
                actual_weights[1] -
                NeutralWeightMultiplier) > 1e-9)
        {

            cerr
                << "\nWARNING:\n";

            cerr
                << "Neutral weight is NOT 4.5 internally.\n";

            cerr
                << "Check MaximumClassWeight in "
                   "linear_svm.hpp.\n";

            return EXIT_FAILURE;
        }

        /*
         * --------------------------------------------------------
         * Training
         * --------------------------------------------------------
         */

        cout
            << "\n";

        cout
            << "========================================\n";

        cout
            << " Training\n";

        cout
            << "========================================\n";

        double best_macro_f1 =
            -numeric_limits<double>::infinity();

        sz epochs_without_improvement = 0;

        LinearSVM best_svm =
            svm;

        // Reuse one sparse feature buffer for the entire training run.
        // This removes repeated vector allocations from the hottest loop.
        SparseVector features;
        features.indices.reserve(128);
        features.values.reserve(128);

        vector<sz> order(train_rows.size());
        iota(order.begin(), order.end(), 0);

        mt19937_64 shuffle_rng(SplitSeed);

        for (
            sz epoch = 1;
            epoch <= Epochs;
            ++epoch)
        {

            const double learning_rate =
                learning_rate_for_epoch(
                    epoch);

            // Shuffle once per epoch for better SGD convergence.
            shuffle(order.begin(), order.end(), shuffle_rng);

            const auto epoch_start =
                chrono::steady_clock::now();

            cout
                << "\n";

            cout
                << "Epoch "
                << epoch
                << "/"
                << Epochs
                << '\n';

            cout
                << "Learning rate : "
                << fixed
                << setprecision(8)
                << learning_rate
                << '\n';

            u64 processed = 0;

            for (sz position = 0; position < order.size(); ++position)
            {

                const Review &review = train_rows[order[position]];

                /*
                 * Reuse the same SparseVector buffers.
                 * TF-IDF remains unigram + bigram with sublinear TF
                 * and L2 normalization.
                 */
                vectorizer.transform_sparse(
                    review.text,
                    features);

                if (features.empty())
                {
                    ++processed;
                    continue;
                }

                svm.train_sample(
                    features,
                    review.sentiment,
                    learning_rate,
                    Regularization);

                ++processed;

                if (
                    processed % 250'000 == 0)
                {

                    cout
                        << "  Processed: "
                        << processed
                        << " / "
                        << train_rows.size()
                        << '\r'
                        << flush;
                }
            }

            cout
                << "\n";

            const auto epoch_end =
                chrono::steady_clock::now();

            const double seconds =
                chrono::duration<double>(
                    epoch_end -
                    epoch_start)
                    .count();

            cout
                << "Epoch time    : "
                << fixed
                << setprecision(2)
                << seconds
                << " sec\n";

            /*
             * ----------------------------------------------------
             * Validation
             * ----------------------------------------------------
             */

            const bool should_validate =
                (epoch %
                 ValidationInterval) == 0 ||
                epoch == Epochs;

            if (
                !should_validate)
            {

                continue;
            }

            cout
                << "\nEvaluating validation set...\n";

            const EvaluationResult validation =
                evaluate_model(
                    svm,
                    vectorizer,
                    validation_rows);

            print_evaluation(
                "Validation",
                validation);

            const double improvement =
                validation.macro_f1 -
                best_macro_f1;

            if (
                improvement >
                MinimumMacroF1Improvement)
            {

                best_macro_f1 =
                    validation.macro_f1;

                best_svm =
                    svm;

                epochs_without_improvement =
                    0;

                cout
                    << "\nNew best model.\n";
            }
            else
            {

                ++epochs_without_improvement;

                cout
                    << "\nNo significant improvement. "
                    << "Patience: "
                    << epochs_without_improvement
                    << "/"
                    << EarlyStoppingPatience
                    << '\n';

                if (
                    epochs_without_improvement >=
                    EarlyStoppingPatience)
                {

                    cout
                        << "\nEarly stopping triggered.\n";

                    break;
                }
            }
        }

        /*
         * --------------------------------------------------------
         * Restore best model
         * --------------------------------------------------------
         */

        svm =
            best_svm;

        cout
            << "\n";

        cout
            << "========================================\n";

        cout
            << " Best Model\n";

        cout
            << "========================================\n";

        cout
            << fixed
            << setprecision(4);

        cout
            << "Best Validation Macro F1 : "
            << best_macro_f1
            << '\n';

        /*
         * --------------------------------------------------------
         * Final Validation
         * --------------------------------------------------------
         */

        const EvaluationResult final_validation =
            evaluate_model(
                svm,
                vectorizer,
                validation_rows);

        print_evaluation(
            "Final Validation",
            final_validation);

        /*
         * --------------------------------------------------------
         * Test
         * --------------------------------------------------------
         */

        cout
            << "\n";

        cout
            << "========================================\n";

        cout
            << " Final Test Evaluation\n";

        cout
            << "========================================\n";

        const EvaluationResult test_result =
            evaluate_model(
                svm,
                vectorizer,
                test_rows);

        print_evaluation(
            "Test",
            test_result);

        /*
         * --------------------------------------------------------
         * Save artifacts
         * --------------------------------------------------------
         */

        const filesystem::path tfidf_path =
            model_directory /
            "tfidf.model";

        const filesystem::path svm_path =
            model_directory /
            "svm.model";

        cout
            << "\nSaving models...\n";

        if (!vectorizer.save(tfidf_path.string()))
        {
            throw runtime_error(
                "Failed to save TF-IDF model: " +
                tfidf_path.string());
        }

        svm.save(svm_path.string());

        cout
            << "TF-IDF model : "
            << tfidf_path
            << '\n';

        cout
            << "SVM model    : "
            << svm_path
            << '\n';

        /*
         * --------------------------------------------------------
         * Final Summary
         * --------------------------------------------------------
         */

        cout
            << "\n";

        cout
            << "========================================\n";

        cout
            << " Training Complete\n";

        cout
            << "========================================\n";

        cout
            << fixed
            << setprecision(4);

        cout
            << "Test Accuracy    : "
            << test_result.accuracy
            << '\n';

        cout
            << "Test Macro F1    : "
            << test_result.macro_f1
            << '\n';

        cout
            << "Test Weighted F1 : "
            << test_result.weighted_f1
            << '\n';

        cout
            << "\nNeutral class:\n";

        cout
            << "  Precision : "
            << test_result.precision[1]
            << '\n';

        cout
            << "  Recall    : "
            << test_result.recall[1]
            << '\n';

        cout
            << "  F1        : "
            << test_result.f1[1]
            << '\n';

        return EXIT_SUCCESS;
    }
    catch (
        const exception &ex)
    {

        cerr
            << "\nERROR: "
            << ex.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
