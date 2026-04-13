/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifdef ENABLE_PROFILING

#include <sycl/sycl.hpp>
#include <sygraph/sycl/event.hpp>
#include <unordered_map>
#include <vector>

namespace sygraph {

namespace detail {

static std::unordered_map<std::string, std::vector<sygraph::Event>> events;
static size_t num_visited_edges = 0;

} // namespace detail

class Profiler {
public:
  static void addEvent(sygraph::Event event, std::string tag = "") {
    if (tag.empty()) { tag = "default"; }
    if (detail::events.find(tag) == detail::events.end()) { detail::events[tag] = std::vector<sygraph::Event>(); }
    detail::events[tag].push_back(event);
  }

  static void addVisitedEdges(size_t visited_edges) { detail::num_visited_edges += visited_edges; }

  static void clear() {
    detail::events.clear();
    detail::num_visited_edges = 0;
  }

  static void print(bool detailed = false) {
    double total_ms = 0.0;
    for (auto& [tag, events] : detail::events) {
      double milliseconds = 0.0;
      size_t i = 0;
      std::cout << " Kernel [" << tag << " x " << events.size() << "] ";
      for (auto& event : events) {
        double event_time = static_cast<double>(event.get_profiling_info<sycl::info::event_profiling::command_end>()
                                                - event.get_profiling_info<sycl::info::event_profiling::command_start>())
                            / 1e6;
        if (detailed) { std::cout << "(" << i++ << ") " << event_time << " ms "; }
        milliseconds += event_time;
      }
      if (detailed) { std::cout << "| "; }
      std::cout << "Time: " << milliseconds << " ms" << std::endl;
      total_ms += milliseconds;
    }
    std::cout << "Total GPU Time: " << total_ms << " ms" << std::endl;
    double mteps = 0.0;
    if (total_ms > 0.0) { mteps = ((detail::num_visited_edges / 1e6) / (total_ms / 1e3)); }
    std::cout << "Total Edge-Througput (MTEPS): " << mteps << " MTEPS" << std::endl;
  }
};

} // namespace sygraph

#endif
