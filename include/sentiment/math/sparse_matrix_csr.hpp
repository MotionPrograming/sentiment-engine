#pragma once

#include "sentiment/core/types.hpp"

#include <limits>
#include <cstdint>

namespace sentiment {

class SparseMatrixCSR {
public:
    using Index = std::uint32_t;

    SparseMatrixCSR() = default;

    SparseMatrixCSR(
        sz rows,
        sz cols
    );

    void reserve(sz non_zero);

    void add(
        Index row,
        Index column,
        double value
    );

    void finalize();

    [[nodiscard]]
    sz rows() const noexcept;

    [[nodiscard]]
    sz cols() const noexcept;

    [[nodiscard]]
    sz non_zero() const noexcept;

    [[nodiscard]]
    double dot_row(
        Index row,
        const vec<double>& weights
    ) const noexcept;

private:
    sz rows_{0};
    sz cols_{0};

    vec<Index> row_ptr_;
    vec<Index> col_indices_;
    vec<double> values_;

    bool finalized_{false};
};

} // namespace sentiment
