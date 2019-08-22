#ifndef STAN_SERVICES_KHAT_HPP
#define STAN_SERVICES_KHAT_HPP

#include <stan/math/prim/mat/fun/mean.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/services/khat.hpp>
#include <Eigen/Dense>
#include <type_traits>

namespace stan {
namespace math {
namespace services {
namespace experimental {
namespace advi {

  template <typename T_x>
  int lx(std::vector<T_x>& a, const Eigen::Matrix<T_x, -1, 1>& x) {
    size_t a_size = a.size();
    size_t x_rows = x.rows();

    std::vector<T_x> a_temp(a_size);
    for (size_t i = 0; i < a_size; ++i)
      a_temp[i] = -a[i];

    std::vector<T_x> k(a_size);
    std::vector<T_x> temp_vec(x_rows);
    
    for (size_t i = 0; i < a_size; ++i) {
      for (size_t ii = 0; ii < x_rows; ++ii) {
        temp_vec[ii] = a_temp[i] * x[ii];
      } 
      for (size_t ii = 0; ii < x_rows; ++ii) {
        temp_vec[ii] = std::log1p(temp_vec[ii]);
      }
      k[i] = mean(temp_vec);
    }

    for (size_t i = 0; i < a_size; ++i) {
      a[i] = std::log(a_temp[i] / k[i]) - k[i] - 1;
    }
    return 0;
  }
  
  template <typename T_x>
  int compute_khat(const std::vector<T_x>& x,
                   const int& min_grid_points = 30,
                   T_x &k = -1.0) {
    size_t N = size_of(x);
    std::vector<T_x> x_s(N);
    x_s = x;
    std::sort(x_s.begin(), x_s.end());

    double prior = 3.0;
    double root_N = std::sqrt(N);
    int M = min_grid_points + std::floor(root_N);
    std::vector<int> jj(M);
    for (size_t i = 0; i < M; ++i)
      jj[i] = i;
    size_t quart1 = std::floor(N / 4 + 0.5);
    auto x_star = x[quart1 - 1];

    std::vector<double> theta(M);
    for(size_t i = 0; i < M; ++i)
      theta[i] = 1 / x_s[i] + (1 - std::sqrt(M / (jj[i] - 0.5))) / (prior * x_star);

    std::vector<T_x> l_theta(M);
    Eigen::Matrix<double, -1, 1> x_lx; x_lx.resize(N, 1);
    for (size_t i = 0; i < N; ++i)
      x_lx << x[i];
    lx(l_theta, x_lx);
    for (size_t i = 0; i < N; ++i)
      l_theta[i] = N * l_theta[i];

    // std::vector<T_x> w_theta(M);
    // for (size_t i = 0; i < M; ++i) {
    //   w_theta[i] = 0;
    //   for (size_t ii = 0; j < M; ++ii)
    //     w_theta[i] = w_theta[i] + exp(l_theta[ii] - l_theta[j]);
    // }

    // T_x theta_hat = 0;
    // for (size_t i = 0; i < M; ++i)
    //   theta_hat = theta_hat + theta[i] * w_theta[i];

    // std::vector<auto> k_vec(N);
    // for (size_t i = 0; i < N; ++i)
    //   k_vec[i] = log(-theta_hat * x[i]);

    // auto k = mean(k_vec);
    // auto sigma = -k / theta_hat;

    // adjust_k_wip(k, N);
    
    // if (std::nan == k)
    //   k = std::numeric_limits<double>::infinity();

    return 0;
  }

  template <typename T_k>
  int adjust_k_wip(T_k& k, const size_t& n) {
    size_t a = 10;
    size_t n_plus_a = n + a;
    k = k * n / n_plus_a + a * 0.5 / n_plus_a;
    return 0;
  }
  

}  // namespace advi
}  // namespace experimental
}  // namespace services
}  // namespace math
}  // namespace stan
#endif
