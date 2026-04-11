#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  sygraph::graph::Properties properties;
  properties.directed = true;
  properties.weighted = true;
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::weighted_directed_5, properties);

  sygraph::algorithms::SSSP sssp(graph);
  uint source = 0;
  sssp.init(source);
  sssp.run();

  std::vector<uint> distances(graph.getVertexCount());
  for (size_t i = 0; i < distances.size(); ++i) { distances[i] = sssp.getDistance(i); }

  sygraph::tests::expectEqual(distances, std::array<uint, 5>{0, 1, 3, 4, 5});
}
