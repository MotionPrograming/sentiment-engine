#include "sentiment/io/dataset_reader.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

struct ParquetDatasetReader::Impl {

    shared_ptr<arrow::RecordBatchReader> batch_reader;

    shared_ptr<arrow::RecordBatch> current_batch;

    shared_ptr<arrow::StringArray> text_array;

    shared_ptr<arrow::Array> sentiment_array;

    sz batch_size{4096};

    sz current_row{0};

    sz total_rows_count{0};

    sz current_batch_row{0};

    bool valid{false};

    bool load_next_batch() {

        current_batch.reset();
        text_array.reset();
        sentiment_array.reset();
        current_batch_row = 0;

        auto result =
            batch_reader->Next();

        if (!result.ok()) {
            throw runtime_error(
                "Failed to read Parquet batch: " +
                result.status().ToString()
            );
        }

        current_batch = *result;

        if (!current_batch) {
            return false;
        }

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
                "Column 'review_text' not found"
            );
        }

        if (!sentiment_column) {
            throw runtime_error(
                "Column 'sentiment' not found"
            );
        }

        text_array =
            static_pointer_cast<arrow::StringArray>(
                text_column
            );

        sentiment_array =
            sentiment_column;

        return true;
    }

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

        auto input =
            *file_result;

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

        unique_ptr<parquet::arrow::FileReader>
            reader = move(*reader_result);

        auto schema_result =
            reader->GetSchema();

        if (!schema_result.ok()) {

            throw runtime_error(
                "Failed to read Parquet schema: " +
                schema_result.status().ToString()
            );
        }

        auto schema =
            *schema_result;

        if (!schema->GetFieldByName(
                "review_text")) {

            throw runtime_error(
                "Required column 'review_text' not found"
            );
        }

        if (!schema->GetFieldByName(
                "sentiment")) {

            throw runtime_error(
                "Required column 'sentiment' not found"
            );
        }

        total_rows_count =
            static_cast<sz>(
                reader->parquet_reader()->metadata()->num_rows()
            );

        auto table_result =
            reader->ReadTable();

        if (!table_result.ok()) {

            throw runtime_error(
                "Failed to initialize Parquet table: " +
                table_result.status().ToString()
            );
        }

        auto table =
            *table_result;

        /*
         * Create batches from Arrow table.
         *
         * This keeps processing bounded by batch_size
         * instead of exposing the whole table to the
         * application pipeline at once.
         */

        auto batches =
            table->CombineChunks();

        if (!batches.ok()) {

            throw runtime_error(
                "Failed to combine Arrow chunks: " +
                batches.status().ToString()
            );
        }

        auto combined_table =
            *batches;

        vector<shared_ptr<arrow::RecordBatch>>
            record_batches;

        auto batch_result =
            combined_table->ToRecordBatches(
                static_cast<int64_t>(batch_size)
            );

        if (!batch_result.ok()) {

            throw runtime_error(
                "Failed to create record batches: " +
                batch_result.status().ToString()
            );
        }

        record_batches =
            *batch_result;

        batch_reader =
            make_shared<
                arrow::RecordBatchReader
            >(
                arrow::RecordBatchReader::Make(
                    record_batches,
                    combined_table->schema()
                )
            );

        /*
         * The exact Arrow factory return differs
         * between Arrow releases. If this constructor
         * fails with your installed Arrow 25 API, we will
         * switch this section to the native
         * parquet RecordBatchReader API.
         */

        valid = true;
    }
};

ParquetDatasetReader::ParquetDatasetReader(
    const str& file_path,
    sz batch_size
)
{
    impl_ =
        make_unique<Impl>(
            file_path,
            batch_size
        );
}

ParquetDatasetReader::~ParquetDatasetReader() = default;

bool ParquetDatasetReader::good() const noexcept {

    return impl_ &&
           impl_->valid;
}

sz ParquetDatasetReader::rows_read() const noexcept {

    return impl_
        ? impl_->current_row
        : 0;
}

sz ParquetDatasetReader::total_rows() const noexcept {

    return impl_
        ? impl_->total_rows_count
        : 0;
}

bool ParquetDatasetReader::next(
    Review& review
) {

    if (!impl_ ||
        !impl_->valid) {

        return false;
    }

    while (true) {

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

        if (impl_->text_array->IsNull(row) ||
            impl_->sentiment_array->IsNull(row)) {

            continue;
        }

        review.text =
            impl_->text_array->GetString(row);

        auto scalar_result =
            impl_->sentiment_array->GetScalar(row);

        if (!scalar_result.ok()) {

            throw runtime_error(
                "Failed to read sentiment value"
            );
        }

        str sentiment =
            (*scalar_result)->ToString();

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

        if (sentiment == "negative") {

            review.sentiment =
                Sentiment::Negative;

        } else if (sentiment == "neutral") {

            review.sentiment =
                Sentiment::Neutral;

        } else if (sentiment == "positive") {

            review.sentiment =
                Sentiment::Positive;

        } else {

            throw runtime_error(
                "Unknown sentiment label: " +
                sentiment
            );
        }

        if (review.text.empty()) {
            continue;
        }

        return true;
    }
}

}