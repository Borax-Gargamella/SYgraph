#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::io::storage::matrices::two_cc);

  sygraph::algorithms::CC cc(graph);
  uint source = 0;
  cc.init(source);
  cc.run();
  cc.reset();

  source = 5;
  cc.init(source);
  cc.run();
}
