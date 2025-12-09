#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
namespace detail {
namespace kernel {

// Lightweight descriptor of the execution range for the advance kernel.
struct LaunchConfig {
  sycl::range<1> global;
  sycl::range<1> local;
  sycl::event dependency;
};

inline size_t roundUpToMultiple(size_t value, size_t factor) {
  if (factor == 0) { return value; }
  return ((value + factor - 1) / factor) * factor;
}

inline size_t ensureLocalMultiple(size_t requested, size_t local_size) {
  if (requested == 0) { return local_size; }
  const size_t rounded = roundUpToMultiple(requested, local_size);
  return std::max(local_size, rounded);
}

} // namespace kernel
} // namespace detail
} // namespace sygraph