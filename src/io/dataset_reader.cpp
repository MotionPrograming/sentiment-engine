#include "sentiment/io/dataset_reader.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

struct ParquetDatasetReader::Impl {

    /*
     * FileReader MUST stay alive while
     * RecordBatchReader is being used.
     */
    unique_ptr<parquet::arrow::FileReader> parquet_reader;

    unique_ptr<arrow::RecordBatchReader> batch_reader;

    shared_ptr<arrow::RecordBatch> current_batch;

    shared_ptr<arrow::StringArray> text_array;

    shared_ptr<arrow::StringArray> sentiment_array;

    sz batch_size{4096};

    sz current_row{0};

    sz total_rows_count{0};

    sz current_batch_row{0};

    bool valid{false};

    // --------------------------------------------------------
    // Load next Arrow RecordBatch
    // --------------------------------------------------------

    bool load_next_batch() {

        current_batch.reset();

        text_array.reset();

        sentiment_array.reset();

        current_batch_row = 0;

        if (!batch_reader) {
            return false;
        }

        auto result =
            batch_reader->Next();

        if (!result.ok()) {

            throw runtime_error(
                "Failed to read Parquet batch: " +
                result.status().ToString()
            );
        }

        current_batch = *result;

        /*
         * nullptr means EOF.
         */
        if (!current_batch) {
            return false;
        }

        // ----------------------------------------------------
        // Find required columns
        // ----------------------------------------------------

        auto text_column =
            current_batch->GetColumnByName(
                "review_text"
            );

        auto sentiment_column =
            current_batch->GetColumnByName(
                "sentiment"
            );

        if (!text_column) {

            throw runtime_error(
                "Column 'review_text' not found in RecordBatch"
            );
        }

        if (!sentiment_column) {

            throw runtime_error(
                "Column 'sentiment' not found in RecordBatch"
            );
        }

        // ----------------------------------------------------
        // Validate text column type
        // ----------------------------------------------------

        if (text_column->type_id() !=
            arrow::Type::STRING) {

            throw runtime_error(
                "Column 'review_text' must be Arrow string"
            );
        }

        text_array =
            static_pointer_cast<
                arrow::StringArray
            >(text_column);

        if (sentiment_column->type_id() !=
            arrow::Type::STRING) {

            throw runtime_error(
                "Column 'sentiment' must be Arrow string"
            );
        }

        sentiment_array =
            static_pointer_cast<
                arrow::StringArray
            >(sentiment_column);

        return true;
    }

    // --------------------------------------------------------
    // Constructor
    // --------------------------------------------------------

    explicit Impl(
        const str& file_path,
        sz requested_batch_size
    )
        : batch_size(
              max<sz>(
                  1,
                  requested_batch_size
              )
          ) {

        // ----------------------------------------------------
        // Open file
        // ----------------------------------------------------

        auto file_result =
            arrow::io::ReadableFile::Open(
                file_path
            );

        if (!file_result.ok()) {

            throw runtime_error(
                "Failed to open Parquet file: " +
                file_result.status().ToString()
            );
        }

        shared_ptr<
            arrow::io::RandomAccessFile
        > input = *file_result;

        // ----------------------------------------------------
        // Open Parquet reader
        // ----------------------------------------------------

        auto reader_result =
            parquet::arrow::OpenFile(
                input,
                arrow::default_memory_pool()
            );

        if (!reader_result.ok()) {

            throw runtime_error(
                "Failed to create Parquet reader: " +
                reader_result.status().ToString()
            );
        }

        parquet_reader =
            move(*reader_result);

        if (!parquet_reader) {

            throw runtime_error(
                "Parquet reader is null"
            );
        }

        // ----------------------------------------------------
        // Configure reader
        // ----------------------------------------------------

        parquet_reader->set_batch_size(
            static_cast<int64_t>(
                batch_size
            )
        );

        /*
         * Enable parallel decoding of columns.
         *
         * Arrow/Parquet handles the internal threading.
         */
        parquet_reader->set_use_threads(true);

        // ----------------------------------------------------
        // Read schema
        // ----------------------------------------------------

        shared_ptr<arrow::Schema> schema;

        auto schema_status =
            parquet_reader->GetSchema(
                &schema
            );

        if (!schema_status.ok()) {

            throw runtime_error(
                "Failed to read Parquet schema: " +
                schema_status.ToString()
            );
        }

        if (!schema) {

            throw runtime_error(
                "Parquet schema is null"
            );
        }

        // ----------------------------------------------------
        // Validate required columns
        // ----------------------------------------------------

        auto text_field =
            schema->GetFieldByName(
                "review_text"
            );

        auto sentiment_field =
            schema->GetFieldByName(
                "sentiment"
            );

        if (!text_field) {

            throw runtime_error(
                "Required column 'review_text' not found"
            );
        }

        if (!sentiment_field) {

            throw runtime_error(
                "Required column 'sentiment' not found"
            );
        }

        // ----------------------------------------------------
        // Validate review_text type
        // ----------------------------------------------------

        if (text_field->type()->id() !=
            arrow::Type::STRING) {

            throw runtime_error(
                "Column 'review_text' must be string"
            );
        }

        // ----------------------------------------------------
        // Total rows
        // ----------------------------------------------------

        total_rows_count =
            static_cast<sz>(
                parquet_reader
                    ->parquet_reader()
                    ->metadata()
                    ->num_rows()
            );

        // ----------------------------------------------------
        // Create RecordBatchReader
        // ----------------------------------------------------

        auto batch_result =
            parquet_reader
                ->GetRecordBatchReader();

        if (!batch_result.ok()) {

            throw runtime_error(
                "Failed to create RecordBatchReader: " +
                batch_result.status().ToString()
            );
        }

        batch_reader =
            move(*batch_result);

        if (!batch_reader) {

            throw runtime_error(
                "RecordBatchReader is null"
            );
        }

        valid = true;
    }
};

// ============================================================
// Constructor
// ============================================================

ParquetDatasetReader::ParquetDatasetReader(
    const str& file_path,
    sz batch_size
) {

    impl_ =
        make_unique<Impl>(
            file_path,
            batch_size
        );
}

// ============================================================
// Destructor
// ============================================================

ParquetDatasetReader::~ParquetDatasetReader() = default;

// ============================================================
// Good
// ============================================================

bool ParquetDatasetReader::good() const noexcept {

    return impl_ &&
           impl_->valid;
}

// ============================================================
// Rows read
// ============================================================

sz ParquetDatasetReader::rows_read() const noexcept {

    return impl_
        ? impl_->current_row
        : 0;
}

// ============================================================
// Total rows
// ============================================================

sz ParquetDatasetReader::total_rows() const noexcept {

    return impl_
        ? impl_->total_rows_count
        : 0;
}

// ============================================================
// Next review
// ============================================================

bool ParquetDatasetReader::next(
    Review& review
) {

    if (!impl_ ||
        !impl_->valid) {

        return false;
    }

    while (true) {

        // ----------------------------------------------------
        // Load next batch when required
        // ----------------------------------------------------

        if (!impl_->current_batch ||
            impl_->current_batch_row >=
                static_cast<sz>(
                    impl_->current_batch->num_rows()
                )) {

            if (!impl_->load_next_batch()) {

                return false;
            }
        }

        const sz row =
            impl_->current_batch_row;

        ++impl_->current_batch_row;

        ++impl_->current_row;

        // ----------------------------------------------------
        // Null values
        // ----------------------------------------------------

        if (impl_->text_array->IsNull(row) ||
            impl_->sentiment_array->IsNull(row)) {

            continue;
        }

        // ----------------------------------------------------
        // Read text
        // ----------------------------------------------------

        review.text =
            impl_->text_array->GetString(row);

        /*
         * Empty reviews are ignored.
         */
        if (review.text.empty()) {

            continue;
        }

        // ----------------------------------------------------
        // Read sentiment scalar
        // ----------------------------------------------------

        str sentiment =
            impl_->sentiment_array->GetString(row);

        // ----------------------------------------------------
        // Normalize label
        // ----------------------------------------------------

        transform(
            sentiment.begin(),
            sentiment.end(),
            sentiment.begin(),
            [](unsigned char c) {

                return static_cast<char>(
                    tolower(c)
                );
            }
        );

        // ----------------------------------------------------
        // Convert to enum
        // ----------------------------------------------------

        if (sentiment == "negative") {

            review.sentiment =
                Sentiment::Negative;
        }
        else if (sentiment == "neutral") {

            review.sentiment =
                Sentiment::Neutral;
        }
        else if (sentiment == "positive") {

            review.sentiment =
                Sentiment::Positive;
        }
        else {

            throw runtime_error(
                "Unknown sentiment label: " +
                sentiment
            );
        }

        return true;
    }
}

} // namespace sentiment
