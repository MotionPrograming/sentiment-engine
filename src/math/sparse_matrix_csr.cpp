#include "sentiment/math/sparse_matrix_csr.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace sentiment {

SparseMatrixCSR::SparseMatrixCSR(
    sz rows,
    sz cols
)
    : rows_(rows),
      cols_(cols),
      row_ptr_(rows + 1, 0) {

    if (rows > static_cast<sz>(
                   std::numeric_limits<Index>::max())) {
        throw std::overflow_error(
            "Too many rows for CSR index type"
        );
    }

    if (cols > static_cast<sz>(
                   std::numeric_limits<Index>::max())) {
        throw std::overflow_error(
            "Too many columns for CSR index type"
        );
    }
}

void SparseMatrixCSR::reserve(sz non_zero) {
    col_indices_.reserve(non_zero);
    values_.reserve(non_zero);
}

void SparseMatrixCSR::add(
    Index row,
    Index column,
    double value
) {
    if (finalized_) {
        throw std::logic_error(
            "Cannot add to finalized CSR matrix"
        );
    }

    if (row >= rows_) {
        throw std::out_of_range("CSR row out of range");
    }

    if (column >= cols_) {
        throw std::out_of_range("CSR column out of range");
    }

    if (value == 0.0) {
        return;
    }

    ++row_ptr_[row + 1];

    col_indices_.push_back(column);
    values_.push_back(value);
}

void SparseMatrixCSR::finalize() {

    if (finalized_) {
        return;
    }

    for (sz i = 1; i < row_ptr_.size(); ++i) {
        row_ptr_[i] += row_ptr_[i - 1];
    }

    finalized_ = true;
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

    if (row >= rows_ ||
        weights.size() < cols_ ||
        !finalized_) {
        return 0.0;
    }

    const Index begin =
        row_ptr_[row];

    const Index end =
        row_ptr_[row + 1];

    double result = 0.0;

    for (Index i = begin; i < end; ++i) {

        result +=
            values_[i] *
            weights[col_indices_[i]];
    }

    return result;
}

} // namespace sentiment
