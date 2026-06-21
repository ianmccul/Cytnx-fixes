#ifndef CYTNX_LINALG_DENSE_MATRIX_INTERNAL_HPP_
#define CYTNX_LINALG_DENSE_MATRIX_INTERNAL_HPP_

#include "Type.hpp"

#include <vector>

namespace cytnx {
  namespace linalg {
    namespace internal {

      template <typename Scalar>
      class DenseMatrix {
       public:
        DenseMatrix() = default;
        DenseMatrix(const cytnx_uint64 rows, const cytnx_uint64 cols)
            : rows_(rows), cols_(cols), data_(rows * cols) {}

        cytnx_uint64 rows() const { return rows_; }
        cytnx_uint64 cols() const { return cols_; }
        cytnx_uint64 size() const { return data_.size(); }

        Scalar &operator()(const cytnx_uint64 row, const cytnx_uint64 col) {
          return data_[row * cols_ + col];
        }

        const Scalar &operator()(const cytnx_uint64 row, const cytnx_uint64 col) const {
          return data_[row * cols_ + col];
        }

       private:
        cytnx_uint64 rows_ = 0;
        cytnx_uint64 cols_ = 0;
        std::vector<Scalar> data_;
      };

    }  // namespace internal
  }  // namespace linalg
}  // namespace cytnx

#endif  // CYTNX_LINALG_DENSE_MATRIX_INTERNAL_HPP_
