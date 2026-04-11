#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);

  sygraph::algorithms::BC bc(graph);
  uint source = 0;
  bc.init(source);
  bc.run();
}
