#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::triangle_3);

  sygraph::algorithms::TC tc(graph);
  tc.init();
  tc.run();

  assert(tc.getNumTriangles() == 1);
}
