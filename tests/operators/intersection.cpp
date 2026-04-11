#include "test_utils.hpp"

int main() {
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_type_t = sygraph::frontier::frontier_type;

  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);

  auto lhs = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);
  auto rhs = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);
  auto out = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);

  for (uint v : {0u, 1u, 3u}) { lhs.insert(v); }
  for (uint v : {1u, 2u, 3u}) { rhs.insert(v); }

  auto event = sygraph::operators::intersection::execute(graph, lhs, rhs, out, [](auto) {});
  event.waitAndThrow();

  sygraph::tests::expectFrontier(out, std::vector<uint>{1, 3});
}
