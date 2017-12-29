#ifndef STAN_LANG_AST_FUN_VAR_DECL_SIZE_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_SIZE_VIS_HPP

#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/array_block_var_decl.hpp>
#include <stan/lang/ast/node/array_fun_var_decl.hpp>
#include <stan/lang/ast/node/array_local_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_corr_block_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_block_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/double_block_var_decl.hpp>
#include <stan/lang/ast/node/double_local_var_decl.hpp>
#include <stan/lang/ast/node/int_block_var_decl.hpp>
#include <stan/lang/ast/node/int_local_var_decl.hpp>
#include <stan/lang/ast/node/matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/matrix_local_var_decl.hpp>
#include <stan/lang/ast/node/ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_local_var_decl.hpp>
#include <stan/lang/ast/node/simplex_block_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_local_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * A visitor for the variant type of variable declarations that
     * returns a vector of size expressions.
     */
    struct var_decl_size_vis
      : public boost::static_visitor<std::vector<expression> > {
      /**
       * Construct a var_decl_size visitor.
       */
      var_decl_size_vis();

      /**
       * Return nil std::vector<expression>
       *
       * @param x variable declaration
       * @return nil std::vector<expression>
       */
      std::vector<expression> operator()(const nil& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const array_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const array_local_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const cholesky_factor_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const cholesky_corr_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const cov_matrix_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const corr_matrix_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const double_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const double_local_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const int_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const int_local_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const matrix_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const matrix_local_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const ordered_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const positive_ordered_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const row_vector_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const row_vector_local_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const simplex_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const unit_vector_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const vector_block_var_decl& x) const;

      /**
       * Return the variable definition.
       *
       * @param x variable declaration
       * @return std::vector<expression> containing definition
       */
      std::vector<expression> operator()(const vector_local_var_decl& x) const;
    };

  }
}
#endif
