#include "sentiment/math/sparse_matrix_csr.hpp"
#include <bits/stdc++.h>

using namespace std;

namespace sentiment {

SparseMatrixCSR::SparseMatrixCSR(
    sz rows,
    sz cols
)
    : rows_(rows),
      cols_(cols),
      row_ptr_(rows + 1, 0) {
}

void SparseMatrixCSR::reserve(
    sz non_zero
) {
    col_indices_.reserve(non_zero);
    values_.reserve(non_zero);
}

void SparseMatrixCSR::add(
    Index row,
    Index column,
    double value
) {
    assert(row < rows_);
    assert(column < cols_);

    ++row_ptr_[row + 1];
    col_indices_.push_back(column);
    values_.push_back(value);
}

void SparseMatrixCSR::finalize() {
    for (sz i = 1; i < row_ptr_.size(); ++i) {
        row_ptr_[i] += row_ptr_[i - 1];
    }
}

sz SparseMatrixCSR::rows() const noexcept {
    return rows_;
}

sz SparseMatrixCSR::cols() const noexcept {
    return cols_;
}

sz SparseMatrixCSR::non_zero() const noexcept {
    return values_.size();
}

double SparseMatrixCSR::dot_row(
    Index row,
    const vec<double>& weights
) const noexcept {
    assert(row < rows_);
    assert(weights.size() >= cols_);

    const Index begin = row_ptr_[row];
    const Index end = row_ptr_[row + 1];

    double result = 0.0;

    for (Index i = begin; i < end; ++i) {
        result += values_[i] * weights[col_indices_[i]];
    }

    return result;
}

} // namespace sentiment