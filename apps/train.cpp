#include "sentiment/io/dataset_reader.hpp"
#include <bits/stdc++.h>


using namespace std;

int main(int argc, char* argv[]) {

    // Dataset path
    const string dataset_path =
        argc > 1
            ? argv[1]
            : "/home/rajeeb/Desktop/software_reviews_3m_.parquet";

    cout << "       SentimentEngine - Dataset Test\n";
    cout << "Dataset: " << dataset_path << "\n\n";

    try {

        // Create Parquet reader
        sentiment::ParquetDatasetReader reader(dataset_path);

        if (!reader.good()) {
            cerr << "Failed to initialize dataset reader.\n";
            return 1;
        }

        cout << "Total rows: " << reader.total_rows() << "\n\n";

        // Read first 10 reviews
        constexpr size_t preview_count = 10;
        sentiment::Review review;
        size_t successful_reads = 0;

        while (successful_reads < preview_count && reader.next(review)) {

            cout << "[" << successful_reads + 1 << "]\n";
            cout << "Text: " << review.text << "\n";
            cout << "Sentiment: " << sentiment::to_string(review.sentiment) << "\n";
            cout << "----------------------------------------\n";

            ++successful_reads;
        }

        // Result
        cout << "\nSuccessfully read: " << successful_reads << " reviews\n";
        cout << "Reader position: " << reader.rows_read() << " / " << reader.total_rows() << "\n";
        cout << "\nDataset reader test completed successfully.\n";

    }
    catch (const exception& ex) {
        cerr << "\nERROR: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}