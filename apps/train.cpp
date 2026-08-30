#include "sentiment/io/dataset_reader.hpp"
#include "sentiment/ml/linear_svm.hpp"
#include "sentiment/text/tfidf_vectorizer.hpp"

#include <bits/stdc++.h>

using namespace std;

int main(int argc, char* argv[]) {

    const string dataset_path =
        "/home/rajeeb/Desktop/sentiment-engine/data/software_reviews_3m.parquet";

    constexpr sentiment::sz max_features = 50'000;

    constexpr double learning_rate = 0.01;
    constexpr double regularization = 0.0001;

    /*
     * Optional:
     *
     * ./sentiment_train
     *     -> full dataset
     *
     * ./sentiment_train 10000
     *     -> first 10,000 rows
     */
    sentiment::sz requested_rows = 0;

    if (argc > 1) {

        try {

            requested_rows =
                static_cast<sentiment::sz>(
                    stoull(argv[1])
                );

            if (requested_rows == 0) {
                cerr << "Row limit must be greater than zero.\n";
                return 1;
            }

        }
        catch (...) {

            cerr << "Usage: ./sentiment_train [max_rows]\n";
            return 1;
        }
    }

    cout << "========================================\n";
    cout << "        SentimentEngine - Training\n";
    cout << "========================================\n";
    cout << "Dataset: " << dataset_path << "\n\n";

    try {

        /*
         * ============================================================
         * PASS 1 — TF-IDF
         * ============================================================
         */

        cout << "[Pass 1/2] Fitting TF-IDF...\n";

        sentiment::ParquetDatasetReader reader(
            dataset_path
        );

        if (!reader.good()) {

            cerr << "Failed to initialize dataset reader.\n";
            return 1;
        }

        const sentiment::sz dataset_rows =
            reader.total_rows();

        const sentiment::sz training_rows =
            requested_rows == 0
                ? dataset_rows
                : min(
                    requested_rows,
                    dataset_rows
                );

        cout << "Dataset rows: "
             << dataset_rows
             << '\n';

        cout << "Rows requested: "
             << training_rows
             << "\n\n";

        sentiment::TfidfVectorizer vectorizer(
            max_features
        );

        sentiment::sz fit_rows = 0;

        vectorizer.fit_stream(
            training_rows,
            [&reader, &fit_rows](
                sentiment::str& document
            ) {

                sentiment::Review review;

                if (!reader.next(review)) {
                    return false;
                }

                document =
                    std::move(review.text);

                ++fit_rows;

                return true;
            }
        );

        cout << "TF-IDF rows consumed: "
             << fit_rows
             << '\n';

        cout << "Vocabulary size: "
             << vectorizer.vocabulary_size()
             << '\n';

        if (vectorizer.vocabulary_size() == 0) {

            cerr << "TF-IDF vocabulary is empty.\n";
            return 1;
        }

        cout << "TF-IDF fitting completed.\n\n";


        /*
         * ============================================================
         * PASS 2 — SVM
         * ============================================================
         */

        cout << "[Pass 2/2] Training Linear SVM...\n";

        sentiment::ParquetDatasetReader training_reader(
            dataset_path
        );

        if (!training_reader.good()) {

            cerr << "Failed to initialize training dataset reader.\n";
            return 1;
        }

        sentiment::LinearSVM svm(
            vectorizer.vocabulary_size()
        );

        sentiment::Review review;

        sentiment::sz processed = 0;
        sentiment::sz skipped = 0;

        const auto start_time =
            chrono::steady_clock::now();

        while (
            processed + skipped < training_rows &&
            training_reader.next(review)
        ) {

            auto features =
                vectorizer.transform_sparse(
                    review.text
                );

            if (!features.valid() ||
                features.indices.empty()) {

                ++skipped;
                continue;
            }

            svm.train_sample(
                features,
                review.sentiment,
                learning_rate,
                regularization
            );

            ++processed;

            if (
                (processed + skipped) % 1'000 == 0
            ) {

                const auto now =
                    chrono::steady_clock::now();

                const auto elapsed =
                    chrono::duration_cast<
                        chrono::seconds
                    >(now - start_time).count();

                cout << "Processed: "
                     << processed
                     << " | skipped: "
                     << skipped
                     << " | elapsed: "
                     << elapsed
                     << "s\n";
            }
        }

        const auto end_time =
            chrono::steady_clock::now();

        const auto elapsed =
            chrono::duration_cast<
                chrono::seconds
            >(end_time - start_time).count();


        /*
         * ============================================================
         * SUMMARY
         * ============================================================
         */

        cout << "\n========================================\n";
        cout << "Training completed\n";
        cout << "========================================\n";

        cout << "Dataset rows: "
             << dataset_rows
             << '\n';

        cout << "Training rows: "
             << training_rows
             << '\n';

        cout << "Processed: "
             << processed
             << '\n';

        cout << "Skipped: "
             << skipped
             << '\n';

        cout << "Vocabulary: "
             << vectorizer.vocabulary_size()
             << '\n';

        cout << "SVM trained: "
             << boolalpha
             << svm.trained()
             << '\n';

        cout << "Training time: "
             << elapsed
             << " seconds\n";

        cout << "Second reader rows consumed: "
             << training_reader.rows_read()
             << '\n';

        cout << "\nTraining pipeline completed successfully.\n";
    }
    catch (const exception& ex) {

        cerr << "\nERROR: "
             << ex.what()
             << '\n';

        return 1;
    }

    return 0;
}
