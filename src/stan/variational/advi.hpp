#ifndef STAN_VARIATIONAL_ADVI_HPP
#define STAN_VARIATIONAL_ADVI_HPP

#include <stan/math.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/compute_effective_sample_size.hpp>
#include <stan/analyze/mcmc/compute_potential_scale_reduction.hpp>
#include <stan/analyze/mcmc/estimate_gpd_params.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/io/dump.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/variational/print_progress.hpp>
#include <stan/variational/families/normal_fullrank.hpp>
#include <stan/variational/families/normal_meanfield.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <ostream>
#include <queue>
#include <string>
#include <vector>
#include <cmath>

namespace stan {

namespace variational {

/**
 * Automatic Differentiation Variational Inference
 *
 * Implements "black box" variational inference using stochastic gradient
 * ascent to maximize the Evidence Lower Bound for a given model
 * and variational family.
 *
 * @tparam Model class of model
 * @tparam Q class of variational distribution
 * @tparam BaseRNG class of random number generator
 */
template <class Model, class Q, class BaseRNG>
class advi {
 public:
  /**
   * Constructor
   *
   * @param[in] m stan model
   * @param[in] cont_params initialization of continuous parameters
   * @param[in,out] rng random number generator
   * @param[in] n_monte_carlo_grad number of samples for gradient computation
   * @param[in] n_monte_carlo_elbo number of samples for ELBO computation
   * @param[in] n_posterior_samples number of samples to draw from posterior
   * @throw std::runtime_error if n_monte_carlo_grad is not positive
   * @throw std::runtime_error if n_monte_carlo_elbo is not positive
   * @throw std::runtime_error if n_posterior_samples is not positive
   */
  advi(Model& m, Eigen::VectorXd& cont_params, BaseRNG& rng,
       int n_monte_carlo_grad, int n_monte_carlo_elbo,
       int n_posterior_samples)
      : model_(m),
        cont_params_(cont_params),
        rng_(rng),
        n_monte_carlo_grad_(n_monte_carlo_grad),
        n_monte_carlo_elbo_(n_monte_carlo_elbo),
        n_posterior_samples_(n_posterior_samples) {
    static const char* function = "stan::variational::advi";
    math::check_positive(function,
                         "Number of Monte Carlo samples for gradients",
                         n_monte_carlo_grad_);
    math::check_positive(function, "Number of Monte Carlo samples for ELBO",
                         n_monte_carlo_elbo_);
    math::check_positive(function, "Number of posterior samples for output",
                         n_posterior_samples_);
  }

  /**
   * Calculates the Evidence Lower BOund (ELBO) by sampling from
   * the variational distribution and then evaluating the log joint,
   * adjusted by the entropy term of the variational distribution.
   *
   * @param[in] variational variational approximation at which to evaluate
   * the ELBO.
   * @param logger logger for messages
   * @return the evidence lower bound.
   * @throw std::domain_error If, after n_monte_carlo_elbo_ number of draws
   * from the variational distribution all give non-finite log joint
   * evaluations. This means that the model is severely ill conditioned or
   * that the variational distribution has somehow collapsed.
   */
  double calc_ELBO(const Q& variational, callbacks::logger& logger) const {
    static const char* function = "stan::variational::advi::calc_ELBO";

    double elbo = 0.0;
    int dim = variational.dimension();
    Eigen::VectorXd zeta(dim);

    int n_dropped_evaluations = 0;
    for (int i = 0; i < n_monte_carlo_elbo_;) {
      variational.sample(rng_, zeta);
      try {
        std::stringstream ss;
        double log_prob = model_.template log_prob<false, true>(zeta, &ss);
        if (ss.str().length() > 0)
          logger.info(ss);
        stan::math::check_finite(function, "log_prob", log_prob);
        elbo += log_prob;
        ++i;
      } catch (const std::domain_error& e) {
        ++n_dropped_evaluations;
        if (n_dropped_evaluations >= n_monte_carlo_elbo_) {
          const char* name = "The number of dropped evaluations";
          const char* msg1 = "has reached its maximum amount (";
          const char* msg2
              = "). Your model may be either severely "
                "ill-conditioned or misspecified.";
          stan::math::throw_domain_error(function, name, n_monte_carlo_elbo_,
                                         msg1, msg2);
        }
      }
    }
    elbo /= n_monte_carlo_elbo_;
    elbo += variational.entropy();
    return elbo;
  }

  /**
   * Calculates the "black box" gradient of the ELBO.
   *
   * @param[in] variational variational approximation at which to evaluate
   * the ELBO.
   * @param[out] elbo_grad gradient of ELBO with respect to variational
   * approximation.
   * @param logger logger for messages
   */
  void calc_ELBO_grad(const Q& variational, Q& elbo_grad,
                      callbacks::logger& logger) const {
    static const char* function = "stan::variational::advi::calc_ELBO_grad";

    stan::math::check_size_match(
        function, "Dimension of elbo_grad", elbo_grad.dimension(),
        "Dimension of variational q", variational.dimension());
    stan::math::check_size_match(
        function, "Dimension of variational q", variational.dimension(),
        "Dimension of variables in model", cont_params_.size());

    variational.calc_grad(elbo_grad, model_, cont_params_, n_monte_carlo_grad_,
                          rng_, logger);
  }

  /**
   * Runs ADVI and writes to output.
   *
   * @param[in] eta eta parameter of stepsize sequence
   * @param[in] max_iterations max number of iterations to run algorithm
   * @param[in] min_window_size Minimum window  size to calculate optimal Rhat
   * @param[in] num_chains Number of VI chains to run
   * @param[in] ess_cut Minimum effective sample size threshold
   * @param[in] mcse_cut MCSE error threshold
   * @param[in] check_frequency Frequency to check for convergence 
   * @param[in] num_grid_points Number of iterate values to calculate min(Rhat)
   * in grid search 
   * @param[in] num_chains Number of VI chains to run
   * @param[in,out] logger logger for messages
   * @param[in,out] parameter_writer writer for parameters (typically to file)
   * @param[in,out] diagnostic_writer writer for diagnostic information
   */
  int run(double eta,
          int max_iterations, int min_window_size, double ess_cut, double mcse_cut, 
          int check_frequency, int num_grid_points, int num_chains, 
          callbacks::logger& logger,
          callbacks::writer& parameter_writer,
          callbacks::writer& diagnostic_writer) const {
    diagnostic_writer("iter,time_in_seconds,ELBO");

    // Initialize variational approximation
    Q variational = Q(cont_params_);

    run_rvi(variational, eta, max_iterations, min_window_size, ess_cut, mcse_cut, 
            check_frequency, num_grid_points, num_chains, logger);

    // Write posterior mean of variational approximations.
    cont_params_ = variational.mean();
    std::vector<double> cont_vector(cont_params_.size());
    for (int i = 0; i < cont_params_.size(); ++i)
      cont_vector.at(i) = cont_params_(i);
    std::vector<int> disc_vector;
    std::vector<double> values;

    std::stringstream msg;
    model_.write_array(rng_, cont_vector, disc_vector, values, true, true,
                       &msg);
    if (msg.str().length() > 0)
      logger.info(msg);

    // The first row of lp_, log_p, and log_g.
    values.insert(values.begin(), {0, 0, 0});
    parameter_writer(values);

    // Draw more from posterior and write on subsequent lines
    logger.info("");
    std::stringstream ss;
    ss << "Drawing a sample of size " << n_posterior_samples_
       << " from the approximate posterior... ";
    logger.info(ss);
    double log_p = 0;
    double log_g = 0;
    // Draw posterior sample. log_g is the log normal densities.
    for (int n = 0; n < n_posterior_samples_; ++n) {
      variational.sample_log_g(rng_, cont_params_, log_g);
      for (int i = 0; i < cont_params_.size(); ++i) {
        cont_vector.at(i) = cont_params_(i);
      }
      std::stringstream msg2;
      model_.write_array(rng_, cont_vector, disc_vector, values, true, true,
                         &msg2);
      //  log_p: Log probability in the unconstrained space
      log_p = model_.template log_prob<false, true>(cont_params_, &msg2);
      if (msg2.str().length() > 0)
        logger.info(msg2);
      // Write lp__, log_p, and log_g.
      values.insert(values.begin(), {0, log_p, log_g});
      parameter_writer(values);
    }
    logger.info("COMPLETED.");
    return stan::services::error_codes::OK;
  }

  /**
  * RVI Diagnostics: Calculates log importance weights
  *
  * @param[in] variational_obj variational family object
  * @param[in, out] weight_vector An Eigen
  * dynamic vector of weights, sorted in descending order
  */
  void lr(const Q& variational_obj, Eigen::VectorXd& weight_vector) 
          const {
    // Need to check the vector is empty
    weight_vector.resize(n_posterior_samples_);
    double log_p, log_g;
    std::stringstream msg2;
    Eigen::VectorXd draws(variational_obj.dimension());
    // Draw posterior sample. log_g is the log normal densities.
    for (int n = 0; n < n_posterior_samples_; ++n) {
      variational_obj.sample_log_g(rng_, draws, log_g);
      //  log_p: Log probability in the unconstrained space
      log_p = model_.template log_prob<false, true>(draws, &msg2);
      weight_vector(n) = log_p - log_g;
    }
    // sort descending order
    std::sort(weight_vector.data(), weight_vector.data() + weight_vector.size(),
              std::greater<double>());
  }
 
  /**
   * RVI Diagnostics
   * Estimate the Effective Sample Size and Monte Carlo Standard Error of posterior samples where
   * MCSE = sqrt( var(parmas) / ess)
   * @param[in] samples An Eigen::VectorXd containing posterior samples @TODO rewrite from here
   * @param[in] ess If specified, will be used as effective sample size instead
   * of calling compute_effective_sample_size()
   * 
   * @return Calculated MCSE
   */
  static double ESS_MCSE(double &ess, double &mcse,
                        const std::vector<const double*> draws,
                        const std::vector<size_t> sizes) {
    int num_chains = sizes.size();
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    if (num_draws < 4) {
      ess = std::numeric_limits<double>::quiet_NaN();
      mcse = std::numeric_limits<double>::quiet_NaN();
      return std::numeric_limits<double>::quiet_NaN();
    }

    // check if chains are constant; all equal to first draw's value
    bool are_all_const = false;
    Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

    for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
              draws[chain_idx], sizes[chain_idx]);

      for (int n = 0; n < num_draws; n++) {
        if (!std::isfinite(draw(n))) {
          ess = std::numeric_limits<double>::quiet_NaN();
          mcse = std::numeric_limits<double>::quiet_NaN();
          return std::numeric_limits<double>::quiet_NaN();
        }
      }

      init_draw(chain_idx) = draw(0);

      if (draw.isApproxToConstant(draw(0))) {
        are_all_const |= true;
      }
    }

    if (are_all_const) {
      // If all chains are constant then return NaN
      // if they all equal the same constant value
      if (init_draw.isApproxToConstant(init_draw(0))) {
        ess = std::numeric_limits<double>::quiet_NaN();;
        mcse = std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
    Eigen::VectorXd chain_mean(num_chains);
    Eigen::VectorXd chain_sq_mean(num_chains);
    Eigen::VectorXd chain_var(num_chains);
    for (int chain = 0; chain < num_chains; ++chain) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
              draws[chain], sizes[chain]);
      stan::analyze::autocovariance<double>(draw, acov(chain));
      chain_mean(chain) = draw.mean();
      chain_sq_mean(chain) = draw.array().square().mean();
      chain_var(chain) = acov(chain)(0) * num_draws / (num_draws - 1);
    }

    double mean_var = chain_var.mean();
    double var_plus = mean_var * (num_draws - 1) / num_draws;
    if (num_chains > 1)
      var_plus += math::variance(chain_mean);
    Eigen::VectorXd rho_hat_s(num_draws);
    rho_hat_s.setZero();
    Eigen::VectorXd acov_s(num_chains);
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(1);
    double rho_hat_even = 1.0;
    rho_hat_s(0) = rho_hat_even;
    double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
    rho_hat_s(1) = rho_hat_odd;

    // Convert raw autocovariance estimators into Geyer's initial
    // positive sequence. Loop only until num_draws - 4 to
    // leave the last pair of autocorrelations as a bias term that
    // reduces variance in the case of antithetical chains.
    size_t s = 1;
    while (s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0) {
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 1);
      rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
      for (int chain = 0; chain < num_chains; ++chain)
        acov_s(chain) = acov(chain)(s + 2);
      rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
      if ((rho_hat_even + rho_hat_odd) >= 0) {
        rho_hat_s(s + 1) = rho_hat_even;
        rho_hat_s(s + 2) = rho_hat_odd;
      }
      s += 2;
    }

    int max_s = s;
    // this is used in the improved estimate, which reduces variance
    // in antithetic case -- see tau_hat below
    if (rho_hat_even > 0)
      rho_hat_s(max_s + 1) = rho_hat_even;

    // Convert Geyer's initial positive sequence into an initial
    // monotone sequence
    for (int s = 1; s <= max_s - 3; s += 2) {
      if (rho_hat_s(s + 1) + rho_hat_s(s + 2) > rho_hat_s(s - 1) + rho_hat_s(s)) {
        rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
        rho_hat_s(s + 2) = rho_hat_s(s + 1);
      }
    }

    double num_total_draws = num_chains * num_draws;
    // Geyer's truncated estimator for the asymptotic variance
    // Improved estimate reduces variance in antithetic case
    double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
    double ess_val = num_total_draws / tau_hat;
    ess = ess_val;
    mcse = std::sqrt((chain_sq_mean.mean() - chain_mean.mean() * chain_mean.mean())/ess_val);
    return 0;
  }

  /**
   * Run Fixed-learning-rate Robust Variational Inference.
   * J. Huggins et al., 2021
   * 
   * @param[in] variational The variational class
   * @param[in] eta Learning rate(stepsize, constant)
   * @param[in] max_runs Max number of VI iterations
   * @param[in] min_window_size Minimum window size to calculate optimal Rhat
   * @param[in] ess_cut Minimum effective sample size threshold
   * @param[in] mcse_cut MCSE error threshold
   * @param[in] check_frequency Frequency to check for convergence 
   * @param[in] num_grid_points Number of iterate values to calculate min(Rhat)
   * in grid search
   * @param[in] num_chains Number of VI chains to run
   * @param[in] logger logger
   */
  void run_rvi(Q& variational, const double eta,
	        const int max_runs, const int min_window_size, int ess_cut,
          const double mcse_cut, int check_frequency, const int num_grid_points,
          const int num_chains, callbacks::logger& logger) const {

    std::stringstream ss;

    if (ess_cut <= 0){
      ess_cut = min_window_size / 8;
    }
    if (check_frequency <= 0){
      check_frequency = min_window_size;
    }

    const int model_dim = variational.dimension();
    const int num_approx_params = variational.num_approx_params();

    std::vector<Q> variational_obj_vec; // vector of variational objs per chain
    std::vector<Q> elbo_grad_vec; // variational to store elbo grads per chain

    // For each chain, save variational parameter values on matrix
    // of dim (n_params, n_iters).
    // tbh, RowMajor isn't necessary, but it made me easier to keep track of
    // indexes and not screw them up
    typedef Eigen::Matrix<double, Eigen::Dynamic, 
                          Eigen::Dynamic, Eigen::RowMajor> histMat;

    std::vector<histMat> hist_vector; // save values per iter per chain

    hist_vector.reserve(num_chains);
    variational_obj_vec.reserve(num_chains);
    elbo_grad_vec.reserve(num_chains);

    for(int i = 0; i < num_chains; i++){
      hist_vector.push_back(histMat(num_approx_params, max_runs));
      variational_obj_vec.push_back(Q(cont_params_));
      elbo_grad_vec.push_back(Q(model_dim));
    }

    // FASO specific variables
    int k_conv = -1; // here -1 is just set to represent 'null' or 'nan'
    int w_check = -1;
    bool success = false;
    int iterations_ran = 0;
    //

    std::chrono::duration<double> optimize_duration(0);
    std::chrono::duration<double> mcse_duration(0);
    logger.info("Start FASO loop\n");
    for (int n_iter = 0; n_iter < max_runs; n_iter++){
      iterations_ran = n_iter;
      auto start_time = std::chrono::steady_clock::now();
      for (int n_chain = 0; n_chain < num_chains; n_chain++){
        calc_ELBO_grad(variational_obj_vec[n_chain], elbo_grad_vec[n_chain], logger);
        variational_obj_vec[n_chain] += eta * elbo_grad_vec[n_chain];
        // stochastic update

        hist_vector[n_chain].col(n_iter) = variational_obj_vec[n_chain].return_approx_params();
      }
      optimize_duration += std::chrono::steady_clock::now() - start_time;
      optimize_duration /= num_chains * n_iter;

      if (k_conv < 0 && n_iter % check_frequency == 0 && n_iter > 0){
        std::stringstream dbg;
        dbg << "Current iteration: " << n_iter << "\n";
        logger.info(dbg);
        double min_chain_rhat = std::numeric_limits<double>::infinity(); // lowest reported rhat value across windows
        for(int grid_i = 0; grid_i < num_grid_points; grid_i++){
          // create equally spaced grid points from min_window_size to 0.95k
          int rhat_lookback_iters = min_window_size + grid_i * 
                                static_cast<int>(0.95 * n_iter - min_window_size) / (num_grid_points - 1);
          
          double rhat, max_rhat = std::numeric_limits<double>::lowest(); // highest rhat across parameters
          for(int k = 0; k < num_approx_params; k++) {
            std::vector<const double*> hist_ptrs;
            std::vector<size_t> chain_length;
            const int split_point = n_iter/2;
            if(num_chains == 1){
              // use split rhat
              chain_length.assign(2, static_cast<size_t>(rhat_lookback_iters / 2));
              hist_ptrs.push_back(hist_vector[0].row(k).data() + n_iter - rhat_lookback_iters + 1);
              hist_ptrs.push_back(hist_ptrs[0] + n_iter - rhat_lookback_iters / 2 + 1);
            }
            else{
              for(int i = 0; i < num_chains; i++){
                //chain_length.push_back(static_cast<size_t>(n_iter * window_size));
                //hist_ptrs.push_back(hist_vector[i].row(k).data());

                // multi-chain split rhat (split each chain into 2)
                chain_length.insert(chain_length.end(), 2, static_cast<size_t>(rhat_lookback_iters / 2));
                hist_ptrs.push_back(hist_vector[i].row(k).data() + n_iter - rhat_lookback_iters + 1);
                hist_ptrs.push_back(hist_vector[i].row(k).data() +  n_iter - rhat_lookback_iters / 2 + 1);
              }
            }
            rhat = stan::analyze::compute_potential_scale_reduction(hist_ptrs, chain_length);
            max_rhat = std::max<double>(max_rhat, rhat);
          }
          if (max_rhat <= min_chain_rhat){
            k_conv = n_iter - rhat_lookback_iters + 1;
            w_check = rhat_lookback_iters;
            min_chain_rhat = max_rhat;
          }
        }
        if (min_chain_rhat > 1.1){
          // if we can't meet the convergence criteria, reset condition variables
          k_conv = -1;
          w_check = -1;
        }
      }

      if (k_conv > 0 && (n_iter - k_conv + 1 == w_check)){
        for(int i = 0; i < num_chains; i++){
          variational_obj_vec[i].set_approx_params(hist_vector[i].block(0, n_iter - w_check, num_approx_params, w_check).rowwise().mean());
          // set parameters per chain to iterate average values
        }

        // pool and average all chain results into single return value
        variational.set_to_zero();
        for(int i = 0; i < num_chains; i++){
          variational += 1.0 / num_chains * variational_obj_vec[i];
        }
        logger.info("checking\n");
        double ess, mcse, min_ess, max_mcse;
        min_ess = std::numeric_limits<double>::infinity(); // min ess across all chains
        max_mcse = std::numeric_limits<double>::lowest(); // max mcse across all chains

        start_time = std::chrono::steady_clock::now();
        logger.info(std::to_string(num_approx_params));
        for(int k = 0; k < num_approx_params; k++){
          std::vector<const double*> hist_ptrs;
          std::vector<size_t> chain_length;
          for(int i = 0; i < num_chains; i++){
            chain_length.push_back(static_cast<size_t>(w_check));
            hist_ptrs.push_back(hist_vector[i].row(k).data() + n_iter - w_check);
          }
          logger.info("start");
          ESS_MCSE(ess, mcse, hist_ptrs, chain_length);
          logger.info("checking??\n");
          if (std::is_same<Q, normal_meanfield>::value && k < model_dim){ // I know, it probably won't work
            // divide MCSE of mu by exp(sigma);
            mcse /= std::exp(hist_vector[0].row(k + model_dim).tail(w_check).mean());
            //sigma corresponding to mu is k + model_dim

            //mcse /= std::exp(hist_vector[0].block(model_dim + k, n_iter - w_check + 1, model_dim + k, w_check).rowwise().mean());
            // TODO: handle multiple chains
            // Currently just uses mean of the first chain
          }
          logger.info("done");
          min_ess = std::min<double>(min_ess, ess);
          max_mcse = std::max<double>(max_mcse, mcse);
        }
        logger.info("checking2\n");
        mcse_duration = std::chrono::steady_clock::now() - start_time;
        mcse_duration /= w_check;
        if (max_mcse < mcse_cut && min_ess >= ess_cut){
          success = true;
          break;
        }
        else{
          w_check *= 1 + 1 / std::sqrt(optimize_duration.count() / mcse_duration.count());
        }
        logger.info("Second finished\n");
      }
    }
    for(int i = 0; i < num_chains; i++){
      variational_obj_vec[i].set_approx_params(hist_vector[i].block(0, iterations_ran - w_check, num_approx_params, w_check).rowwise().mean());
      // set parameters per chain to iterate average values
    }

    // pool and average all chain results into single return value
    variational.set_to_zero();
    for(int i = 0; i < num_chains; i++){
      variational += 1.0 / num_chains * variational_obj_vec[i];
    }

    ss << "k_conv: " << k_conv << " total iterations ran: " << iterations_ran << "\n";
    if (success){
      ss << "Optimization finished succesfully\n";
    }
    else{
      ss << "Optimization failed. Results are probably unreliable.\n";
    }
    for(int i = 0; i < num_chains; i++){
      ss << "Chain " << i << " mean:\n" << variational_obj_vec[i].mean() << "\n";
    }
    ss << "----\nQ variational:\n" << variational.mean() << "\n----\n";
    ss << "Num of Model params: " << model_dim << "\n";
    ss << "Num of Approx params: " << num_approx_params << "\n";
    logger.info(ss);
  }

 protected:
  Model& model_;
  Eigen::VectorXd& cont_params_;
  BaseRNG& rng_;
  int n_monte_carlo_grad_;
  int n_monte_carlo_elbo_;
  int n_posterior_samples_;
};
}  // namespace variational
}  // namespace stan
#endif
