#pragma once

#include "sentiment/core/types.hpp"

#include <memory>

namespace sentiment {

// Forward declaration if not defined in types.hpp
struct Review;

class ParquetDatasetReader {

public:

    explicit ParquetDatasetReader(
        const str& file_path,
        sz batch_size = 4096
    );

    ~ParquetDatasetReader();

    ParquetDatasetReader(
        const ParquetDatasetReader&
    ) = delete;

    ParquetDatasetReader& operator=(
        const ParquetDatasetReader&
    ) = delete;

    [[nodiscard]]
    bool good() const noexcept;

    [[nodiscard]]
    bool next(
        Review& review
    );

    [[nodiscard]]
    sz rows_read() const noexcept;

    [[nodiscard]]
    sz total_rows() const noexcept;

private:

    struct Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace sentiment