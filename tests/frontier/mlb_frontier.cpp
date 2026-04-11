#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::star_5);

  auto frontier =
      sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::mlb>(q, graph);

  frontier.insert(1);
  frontier.insert(3);
  frontier.insert(4);
  sygraph::tests::expectFrontier(frontier, std::vector<uint>{1, 3, 4});

  auto saved = frontier.saveState();

  frontier.remove(3);
  sygraph::tests::expectFrontier(frontier, std::vector<uint>{1, 4});

  frontier.clear();
  assert(frontier.empty());

  frontier.loadState(saved);
  sygraph::tests::expectFrontier(frontier, std::vector<uint>{1, 3, 4});
}
