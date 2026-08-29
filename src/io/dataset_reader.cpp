#include <stdexcept>
#include "sentiment/io/dataset_reader.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

struct ParquetDatasetReader::Impl {
    shared_ptr<arrow::Table> table;

    shared_ptr<arrow::ChunkedArray> text_column;
    shared_ptr<arrow::ChunkedArray> sentiment_column;

    sz current_row{0};
    sz total_rows{0};

    // Cache current chunk state to avoid linear scans
    sz current_chunk_idx{0};
    sz chunk_offset{0};

    shared_ptr<arrow::StringArray> current_text_array;
    shared_ptr<arrow::Array> current_sentiment_array;

    bool valid{false};

    explicit Impl(const str& file_path) {
        // 1. Open Parquet file through Arrow filesystem layer
        auto file_result = arrow::io::ReadableFile::Open(file_path);
        if (!file_result.ok()) {
            throw runtime_error("Failed to open Parquet file: " + file_result.status().ToString());
        }

        shared_ptr<arrow::io::RandomAccessFile> input = *file_result;

        // 2. Create Parquet Arrow reader
        auto reader_result = parquet::arrow::OpenFile(input, arrow::default_memory_pool());
        if (!reader_result.ok()) {
            throw runtime_error("Failed to create Parquet reader: " + reader_result.status().ToString());
        }

        unique_ptr<parquet::arrow::FileReader> reader = move(*reader_result);

        // 3. Read the table
        auto table_result = reader->ReadTable();
        if (!table_result.ok()) {
            throw runtime_error("Failed to read Parquet table: " + table_result.status().ToString());
        }

        table = *table_result;

        // 4. Locate required columns
        text_column = table->GetColumnByName("review_text");
        sentiment_column = table->GetColumnByName("sentiment");

        if (!text_column) throw runtime_error("Required column 'review_text' not found");
        if (!sentiment_column) throw runtime_error("Required column 'sentiment' not found");

        total_rows = static_cast<sz>(table->num_rows());

        if (total_rows > 0 && text_column->num_chunks() > 0) {
            load_chunk(0);
        }

        valid = true;
    }

    void load_chunk(sz chunk_idx) {
        current_chunk_idx = chunk_idx;
        current_text_array = static_pointer_cast<arrow::StringArray>(text_column->chunk(chunk_idx));
        current_sentiment_array = sentiment_column->chunk(chunk_idx);
    }
};

ParquetDatasetReader::ParquetDatasetReader(
    const str& file_path,
    sz batch_size
) {
    (void)batch_size;
    impl_ = make_unique<Impl>(file_path);
}

ParquetDatasetReader::~ParquetDatasetReader() = default;

bool ParquetDatasetReader::good() const noexcept {
    return impl_ && impl_->valid;
}

sz ParquetDatasetReader::rows_read() const noexcept {
    return impl_ ? impl_->current_row : 0;
}

sz ParquetDatasetReader::total_rows() const noexcept {
    return impl_ ? impl_->total_rows : 0;
}

bool ParquetDatasetReader::next(Review& review) {
    if (!impl_ || !impl_->valid || impl_->current_row >= impl_->total_rows) {
        return false;
    }

    // Check if we need to advance to the next Chunk
    while (impl_->current_text_array && 
           (impl_->current_row - impl_->chunk_offset) >= static_cast<sz>(impl_->current_text_array->length())) {
        
        impl_->chunk_offset += impl_->current_text_array->length();
        impl_->current_chunk_idx++;

        if (impl_->current_chunk_idx < static_cast<sz>(impl_->text_column->num_chunks())) {
            impl_->load_chunk(impl_->current_chunk_idx);
        } else {
            return false;
        }
    }

    const sz local_idx = impl_->current_row - impl_->chunk_offset;

    // Direct O(1) Memory Access
    if (impl_->current_text_array->IsNull(local_idx) || impl_->current_sentiment_array->IsNull(local_idx)) {
        ++impl_->current_row;
        return false;
    }

    // Read Review Text
    review.text = impl_->current_text_array->GetString(local_idx);

    // Read Sentiment Label
    str sentiment = impl_->current_sentiment_array->GetScalar(local_idx).ValueOrDie()->ToString();

    for (char& c : sentiment) {
        c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }

    if (sentiment == "negative") {
        review.sentiment = Sentiment::Negative;
    } else if (sentiment == "neutral") {
        review.sentiment = Sentiment::Neutral;
    } else if (sentiment == "positive") {
        review.sentiment = Sentiment::Positive;
    } else {
        throw runtime_error("Unknown sentiment label: " + sentiment);
    }

    ++impl_->current_row;
    return true;
}

} // namespace sentiment