#ifndef STAN_LANG_AST_CORR_MATRIX_BLOCK_TYPE_HPP
#define STAN_LANG_AST_CORR_MATRIX_BLOCK_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    /**
     * Correlation matrix block var type.
     */
    struct corr_matrix_block_type {
      /**
       * Number of rows and columns
       */
      expression K_;

      /**
       * Construct a block var type with default values.
       */
      corr_matrix_block_type();

      /**
       * Construct a block var type with specified values.
       *
       * @param K corr matrix size
       */
      corr_matrix_block_type(const expression& K);
    };

  }
}
#endif
