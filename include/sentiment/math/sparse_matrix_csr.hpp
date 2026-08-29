#include "sentiment/core/types.hpp"

#pragma once

#include <bits/stdc++.h>

namespace sentiment {

using u32 = std::uint32_t;
using sz  = std::size_t;

template <typename T>
using vec = std::vector<T>;

class SparseMatrixCSR {
public:
    using Index = u32;

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
};

} // namespace sentiment