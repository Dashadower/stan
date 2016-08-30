#include <gtest/gtest.h>
#include <test/test-models/performance/twoCmtOral.hpp>
#include <stan/version.hpp>
#include <ctime>
#include <test/performance/utility.hpp>
#include <boost/algorithm/string/trim.hpp>

class performance : public ::testing::Test {
public:
  static void SetUpTestCase() {
    N = 1;
    seconds_per_run.resize(N);
    last_draws_per_run.resize(N);
    matches_tagged_version = false;
    all_values_same = false;
  }

  static int N;
  static std::vector<double> seconds_per_run;
  static std::vector<std::vector<double> > last_draws_per_run;
  static bool matches_tagged_version;
  static bool all_values_same;
};

int performance::N;
std::vector<double> performance::seconds_per_run;
std::vector<std::vector<double> > performance::last_draws_per_run;
bool performance::matches_tagged_version;
bool performance::all_values_same;


using stan::test::performance::run_command_output;
using stan::test::performance::run_command;
using stan::test::performance::get_last_iteration_from_file;
using stan::test::performance::quote;
using stan::test::performance::get_git_hash;
using stan::test::performance::get_git_date;
using stan::test::performance::get_date;

TEST_F(performance, run) {
  clock_t t;
  for (int n = 0; n < N; ++n) {
    std::cout << "iteration: " << n << " / " << N << std::endl;
    t = clock();      // start timer
    stan::test::performance::command<stan_model>(1000,
                                                 10000,
                                                 "src/test/test-models/performance/twoCmtOral.data.R",
                                                 "test/performance/ode_twoCmtOral_output.csv",
                                                 0U);
    t = clock() - t;  // end timer
    seconds_per_run[n] = static_cast<double>(t) / CLOCKS_PER_SEC;
    last_draws_per_run[n]
      = get_last_iteration_from_file("test/performance/od_twoCmtOral_output.csv");
  }
  SUCCEED();
}

// // evaluate
// TEST_F(performance, values_from_tagged_version) {
//   int N_values = 9;
//   ASSERT_EQ(N_values, last_draws_per_run[0].size())
//     << "last tagged version, 2.9.0, had " << N_values << " elements";

//   std::vector<double> first_run = last_draws_per_run[0];
//   EXPECT_FLOAT_EQ(-65.742996, first_run[0])
//     << "lp__: index 0";

//   EXPECT_FLOAT_EQ(0.622922, first_run[1])
//     << "accept_stat__: index 1";

//   EXPECT_FLOAT_EQ(1.15558, first_run[2])
//     << "stepsize__: index 2";

//   EXPECT_FLOAT_EQ(1, first_run[3])
//     << "treedepth__: index 3";

//   EXPECT_FLOAT_EQ(3, first_run[4])
//     << "n_leapfrog__: index 4";

//   EXPECT_FLOAT_EQ(0, first_run[5])
//     << "divergent__: index 5";

//   EXPECT_FLOAT_EQ(67.453796, first_run[6])
//     << "energy__: index 6";

//   EXPECT_FLOAT_EQ(1.40756, first_run[7])
//     << "beta.1: index 7";

//   EXPECT_FLOAT_EQ(-0.763035, first_run[8])
//     << "beta.2: index 8";

//   matches_tagged_version = !HasNonfatalFailure();
// }

// TEST_F(performance, values_same_run_to_run) {
//   int N_values = last_draws_per_run[0].size();

//   for (int i = 0; i < N_values; i++) {
//     double expected_value = last_draws_per_run[0][i];
//     for (int n = 1; n < N; n++) {
//       EXPECT_FLOAT_EQ(expected_value, last_draws_per_run[n][i])
//         << "expecting run to run values to be the same. Found run "
//         << n << " to have different values than the 0th run for "
//         << "index: " << i;
//     }
//   }
//   all_values_same = !HasNonfatalFailure();
// }

// TEST_F(performance, check_output_is_same) {
//   std::ifstream file_stream;
//   file_stream.open("test/performance/logistic_output.csv",
//                    std::ios_base::in);
//   ASSERT_TRUE(file_stream.good());

//   std::string line, expected;

//   getline(file_stream, line);
//   expected = "# stan_version_major = " + stan::MAJOR_VERSION;
//   ASSERT_EQ(expected, line);
//   ASSERT_TRUE(file_stream.good());


//   getline(file_stream, line);
//   expected = "# stan_version_minor = " + stan::MINOR_VERSION;
//   ASSERT_EQ(expected, line);
//   ASSERT_TRUE(file_stream.good());

//   getline(file_stream, line);
//   expected = "# stan_version_patch = " + stan::PATCH_VERSION;
//   ASSERT_EQ(expected, line);
//   ASSERT_TRUE(file_stream.good());

//   getline(file_stream, line);
//   ASSERT_EQ("# model = logistic_model", line);
//   ASSERT_TRUE(file_stream.good());

//   getline(file_stream, line);
//   ASSERT_EQ("lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__,beta.1,beta.2", line);
//   ASSERT_TRUE(file_stream.good());

//   getline(file_stream, line);
//   ASSERT_EQ("# Adaptation terminated", line);
//   ASSERT_TRUE(file_stream.good());

//   file_stream.close();
// }

// TEST_F(performance, write_results_to_disk) {
//   std::stringstream header;
//   std::stringstream line;

//   // current date / time
//   header << quote("date");
//   line << quote(get_date());

//   // git hash
//   header << "," << quote("git hash") << "," << quote("git date");
//   line << "," << quote(get_git_hash()) << "," << quote(get_git_date());

//   // model name: "logistic"
//   header << "," << quote("model name");
//   line << "," << quote("logistic");

//   // matches tagged values
//   header << "," << quote("matches tagged version");
//   line << "," << quote(matches_tagged_version ? "yes" : "no");

//   // all values same
//   header << "," << quote("all values same");
//   line << "," << quote(all_values_same ? "yes" : "no");

//   // N times
//   for (int n = 0; n < N; n++) {
//     std::stringstream ss;
//     ss << "run " << n+1;
//     header << "," << quote(ss.str());
//     line << "," << seconds_per_run[n];
//   }


//   // append output to: test/performance/performance.csv
//   bool write_header = false;
//   std::fstream file_stream;

//   file_stream.open("test/performance/performance.csv",
//                    std::ios_base::in);
//   if (file_stream.peek() == std::fstream::traits_type::eof()) {
//     write_header = true;
//   } else {
//     std::string file_header;
//     std::getline(file_stream, file_header);

//     EXPECT_EQ(file_header, header.str())
//       << "header of file is different";
//     if (file_header != header.str())
//       write_header = true;
//   }
//   file_stream.close();

//   file_stream.open("test/performance/performance.csv",
//                    std::ios_base::app);
//   if (write_header)
//     file_stream << header.str() << std::endl;
//   file_stream << line.str() << std::endl;
//   file_stream.close();
// }