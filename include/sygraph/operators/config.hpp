/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace sygraph {
namespace operators {

enum class load_balancer {
  workitem_mapped,
  subgroup_mapped,
  workgroup_mapped,
};

enum class direction {
  push,
  pull,
};

namespace frontier_size {

constexpr int fetch_from_memory = -1; // Fetch the frontier size from memory
constexpr int infer_from_device = 0;  // Infer the frontier size from the device

} // namespace frontier_size

} // namespace operators
} // namespace sygraph