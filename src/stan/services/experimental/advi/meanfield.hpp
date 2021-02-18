#ifndef STAN_SERVICES_EXPERIMENTAL_ADVI_MEANFIELD_HPP
#define STAN_SERVICES_EXPERIMENTAL_ADVI_MEANFIELD_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/experimental_message.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/io/var_context.hpp>
#include <stan/variational/advi.hpp>
#include <boost/random/additive_combine.hpp>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace experimental {
namespace advi {

/**
 * Runs mean field ADVI.
 *
 * @tparam Model A model implementation
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] grad_samples number of samples for Monte Carlo estimate
 *   of gradients
 * @param[in] elbo_samples number of samples for Monte Carlo estimate
 *   of ELBO
 * @param[in] max_iterations maximum number of iterations
 * @param[in] eta stepsize scaling parameter for variational inference
 * @param[in] min_window_size Minimum size of window to produce optimal window size
 * @param[in] check_frequency Frequency to check for convergence
 * @param[in] num_grid_points Number of iterate values to calculate min(Rhat)
 * @param[in] mcse_cut MCSE termination criteria
 * @param[in] ess_cut effective sample size termination criteria, equals min_sample_size
 * @param[in] num_chains Number of VI chains to run
 * @param[in] output_samples number of posterior samples to draw and
 *   save
 * @param[in,out] interrupt callback to be called every iteration
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] parameter_writer output for parameter values
 * @param[in,out] diagnostic_writer output for diagnostic values
 * @return error_codes::OK if successful
 */
template <class Model>
int meanfield(Model& model, const stan::io::var_context& init,
              unsigned int random_seed, unsigned int chain, double init_radius,
              int grad_samples, int elbo_samples, int max_iterations,
              double eta, int min_window_size, int check_frequency, 
              int num_grid_points, double mcse_cut, double ess_cut,
              int num_chains, int output_samples,
              callbacks::interrupt& interrupt, callbacks::logger& logger,
              callbacks::writer& init_writer,
              callbacks::writer& parameter_writer,
              callbacks::writer& diagnostic_writer) {
  util::experimental_message(logger);

  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector = util::initialize(
      model, init, rng, init_radius, true, logger, init_writer);

  std::vector<std::string> names;
  names.push_back("lp__");
  names.push_back("log_p__");
  names.push_back("log_g__");
  model.constrained_param_names(names, true, true);
  parameter_writer(names);

  Eigen::VectorXd cont_params
      = Eigen::Map<Eigen::VectorXd>(&cont_vector[0], cont_vector.size(), 1);

  stan::variational::advi<Model, stan::variational::normal_meanfield,
                          boost::ecuyer1988>
    cmd_advi(model, cont_params, rng, grad_samples, elbo_samples,
	     output_samples);
  cmd_advi.run(eta, max_iterations, min_window_size, ess_cut, mcse_cut,
               check_frequency, num_grid_points,  num_chains, logger, 
               parameter_writer, diagnostic_writer);

  return 0;
}
}  // namespace advi
}  // namespace experimental
}  // namespace services
}  // namespace stan
#endif
