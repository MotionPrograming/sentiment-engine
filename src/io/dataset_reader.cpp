#include "sentiment/io/dataset_reader.hpp"
#include "sentiment/core/types.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sentiment {

struct ParquetDatasetReader::Impl {

    std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
    std::unique_ptr<arrow::RecordBatchReader> batch_reader;

    std::shared_ptr<arrow::RecordBatch> current_batch;
    std::shared_ptr<arrow::StringArray> text_array;
    std::shared_ptr<arrow::StringArray> sentiment_string_array;
    std::shared_ptr<arrow::UInt8Array> sentiment_uint8_array;

    sz batch_size{4096};
    sz current_row{0};
    sz total_rows_count{0};
    sz current_batch_row{0};

    enum class SentimentType {
        String,
        UInt8
    };

    SentimentType sentiment_type = SentimentType::String;
    bool valid{false};

    bool load_next_batch() {
        current_batch.reset();
        text_array.reset();
        sentiment_string_array.reset();
        sentiment_uint8_array.reset();

        current_batch_row = 0;

        if (!batch_reader) {
            return false;
        }

        auto result = batch_reader->Next();
        if (!result.ok()) {
            throw std::runtime_error(
                "Failed to read Parquet batch: " + result.status().ToString()
            );
        }

        current_batch = *result;
        if (!current_batch) {
            return false;
        }

        auto text_column = current_batch->GetColumnByName("review_text");
        auto sentiment_column = current_batch->GetColumnByName("sentiment");

        if (!text_column) {
            throw std::runtime_error("Column 'review_text' not found");
        }
        if (!sentiment_column) {
            throw std::runtime_error("Column 'sentiment' not found");
        }

        if (text_column->type_id() != arrow::Type::STRING) {
            throw std::runtime_error("Column 'review_text' must be Arrow string");
        }

        text_array = std::static_pointer_cast<arrow::StringArray>(text_column);

        if (sentiment_type == SentimentType::String) {
            if (sentiment_column->type_id() != arrow::Type::STRING) {
                throw std::runtime_error("Column 'sentiment' must be Arrow string");
            }
            sentiment_string_array =
                std::static_pointer_cast<arrow::StringArray>(sentiment_column);
        } else {
            if (sentiment_column->type_id() != arrow::Type::UINT8) {
                throw std::runtime_error("Column 'sentiment' must be UInt8");
            }
            sentiment_uint8_array =
                std::static_pointer_cast<arrow::UInt8Array>(sentiment_column);
        }

        return true;
    }

    static Sentiment parse_sentiment(std::string sentiment_str) {
        std::transform(
            sentiment_str.begin(),
            sentiment_str.end(),
            sentiment_str.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );

        if (sentiment_str == "negative") return Sentiment::Negative;
        if (sentiment_str == "neutral")  return Sentiment::Neutral;
        if (sentiment_str == "positive") return Sentiment::Positive;

        throw std::runtime_error("Unknown sentiment label: " + sentiment_str);
    }

    explicit Impl(const str& file_path, sz requested_batch_size)
        : batch_size(std::max<sz>(1, requested_batch_size)) {

        auto file_result = arrow::io::ReadableFile::Open(file_path);
        if (!file_result.ok()) {
            throw std::runtime_error(
                "Failed to open Parquet file: " + file_result.status().ToString()
            );
        }

        std::shared_ptr<arrow::io::RandomAccessFile> input = *file_result;

        auto reader_result = parquet::arrow::FileReader::Make(
            arrow::default_memory_pool(),
            parquet::ParquetFileReader::Open(input)
        );

        if (!reader_result.ok()) {
            throw std::runtime_error(
                "Failed to create Parquet reader: " + reader_result.status().ToString()
            );
        }

        parquet_reader = std::move(*reader_result);

        if (!parquet_reader) {
            throw std::runtime_error("Parquet reader is null");
        }

        parquet_reader->set_batch_size(static_cast<int64_t>(batch_size));
        parquet_reader->set_use_threads(true);

        std::shared_ptr<arrow::Schema> schema;
        auto schema_status = parquet_reader->GetSchema(&schema);

        if (!schema_status.ok() || !schema) {
            throw std::runtime_error("Failed to read Parquet schema");
        }

        auto text_field = schema->GetFieldByName("review_text");
        auto sentiment_field = schema->GetFieldByName("sentiment");

        if (!text_field || !sentiment_field) {
            throw std::runtime_error("Required column 'review_text' or 'sentiment' not found");
        }

        if (text_field->type()->id() != arrow::Type::STRING) {
            throw std::runtime_error("Column 'review_text' must be string");
        }

        if (sentiment_field->type()->id() == arrow::Type::STRING) {
            sentiment_type = SentimentType::String;
        } else if (sentiment_field->type()->id() == arrow::Type::UINT8) {
            sentiment_type = SentimentType::UInt8;
        } else {
            throw std::runtime_error("Column 'sentiment' must be string or uint8");
        }

        total_rows_count = static_cast<sz>(
            parquet_reader->parquet_reader()->metadata()->num_rows()
        );

        auto batch_result = parquet_reader->GetRecordBatchReader();
        if (!batch_result.ok()) {
            throw std::runtime_error(
                "Failed to create RecordBatchReader: " + batch_result.status().ToString()
            );
        }

        batch_reader = std::move(*batch_result);
        if (!batch_reader) {
            throw std::runtime_error("RecordBatchReader is null");
        }

        valid = true;
    }
};

// Member definitions placed outside struct Impl

ParquetDatasetReader::ParquetDatasetReader(const str& file_path, sz batch_size) {
    impl_ = std::make_unique<Impl>(file_path, batch_size);
}

ParquetDatasetReader::~ParquetDatasetReader() = default;

bool ParquetDatasetReader::good() const noexcept {
    return impl_ && impl_->valid;
}

sz ParquetDatasetReader::rows_read() const noexcept {
    return impl_ ? impl_->current_row : 0;
}

sz ParquetDatasetReader::total_rows() const noexcept {
    return impl_ ? impl_->total_rows_count : 0;
}

bool ParquetDatasetReader::next(Review& review) {
    if (!impl_ || !impl_->valid) {
        return false;
    }

    while (true) {
        if (!impl_->current_batch ||
            impl_->current_batch_row >= static_cast<sz>(impl_->current_batch->num_rows())) {
            if (!impl_->load_next_batch()) {
                return false;
            }
        }

        const sz row = impl_->current_batch_row;
        ++impl_->current_batch_row;
        ++impl_->current_row;

        if (impl_->text_array->IsNull(row)) {
            continue;
        }

        if (impl_->sentiment_type == Impl::SentimentType::String) {
            if (impl_->sentiment_string_array->IsNull(row)) {
                continue;
            }
        } else {
            if (impl_->sentiment_uint8_array->IsNull(row)) {
                continue;
            }
        }

        review.text = impl_->text_array->GetString(row);
        if (review.text.empty()) {
            continue;
        }

        if (impl_->sentiment_type == Impl::SentimentType::String) {
            review.sentiment = Impl::parse_sentiment(
                impl_->sentiment_string_array->GetString(row)
            );
        } else {
            const u8 value = impl_->sentiment_uint8_array->Value(row);
            if (value > 2) {
                throw std::runtime_error("Invalid numeric sentiment label");
            }
            review.sentiment = static_cast<Sentiment>(value);
        }

        return true;
    }
}

} // namespace sentiment