#pragma once

#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

namespace sygraph::tests {

namespace fixtures {

inline constexpr std::string_view line_5 = "5\n"
                                           "0 1 0 0 0\n"
                                           "1 0 1 0 0\n"
                                           "0 1 0 1 0\n"
                                           "0 0 1 0 1\n"
                                           "0 0 0 1 0";

inline constexpr std::string_view star_5 = "5\n"
                                           "0 1 1 1 1\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0";

inline constexpr std::string_view triangle_3 = "3\n"
                                               "0 1 1\n"
                                               "1 0 1\n"
                                               "1 1 0";

inline constexpr std::string_view weighted_directed_5 = "5\n"
                                                        "0 1 4 0 0\n"
                                                        "0 0 2 6 0\n"
                                                        "0 0 0 1 5\n"
                                                        "0 0 0 0 1\n"
                                                        "0 0 0 0 0";

} // namespace fixtures

inline sycl::queue makeQueue() {
  setenv("UR_ADAPTERS_FORCE_LOAD", "opencl", 0);
  try {
    return sycl::queue{sycl::gpu_selector_v};
  } catch (const sycl::exception&) {
    try {
      return sycl::queue{sycl::default_selector_v};
    } catch (const sycl::exception&) {
      std::cout << "Skipping test: no SYCL platform available" << std::endl;
      std::exit(0);
    }
  }
}

template<sygraph::memory::space Space = sygraph::memory::space::shared, typename ValueT = uint, typename IndexT = uint, typename OffsetT = uint>
auto buildGraphFromMatrix(sycl::queue& q, std::string_view matrix, sygraph::graph::Properties properties = {}) {
  std::istringstream iss{std::string(matrix)};
  auto csr = sygraph::io::csr::fromMatrix<ValueT, IndexT, OffsetT>(iss);
  return sygraph::graph::build::fromCSR<Space>(q, std::move(csr), properties);
}

template<typename T, size_t N>
void expectEqual(const std::vector<T>& actual, const std::array<T, N>& expected) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) { assert(actual[i] == expected[i]); }
}

template<typename T>
void expectEqual(const std::vector<T>& actual, const std::vector<T>& expected) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) { assert(actual[i] == expected[i]); }
}

template<typename FrontierT>
std::vector<typename FrontierT::type_t> activeElements(const FrontierT& frontier) {
  using value_t = typename FrontierT::type_t;

  std::vector<value_t> values;
  for (size_t i = 0; i < frontier.getNumElems(); ++i) {
    if (frontier.check(i)) { values.push_back(static_cast<value_t>(i)); }
  }
  return values;
}

template<typename FrontierT>
void expectFrontier(const FrontierT& frontier, const std::vector<typename FrontierT::type_t>& expected) {
  expectEqual(activeElements(frontier), expected);
}

} // namespace sygraph::tests
