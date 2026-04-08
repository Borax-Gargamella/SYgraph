#include <sstream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

namespace {

template<sygraph::memory::space Space>
void buildAndDestroyGraphs(sycl::queue& q) {
  auto matrix = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(matrix.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);

  {
    auto undirected = sygraph::graph::build::fromCSR<Space>(q, csr);
    assert(undirected.getVertexCount() == csr.getRowOffsetsSize());
  }

  sygraph::graph::Properties directed_props;
  directed_props.directed = true;
  {
    auto directed = sygraph::graph::build::fromCSR<Space>(q, csr, directed_props);
    assert(directed.getEdgeCount() == csr.getNumNonzeros());
  }

  q.wait_and_throw();
}

} // namespace

int main() {
#ifndef GENERATE_SAMPLE_DATA
  return 0;
#else
  sycl::queue q{sycl::gpu_selector_v};

  buildAndDestroyGraphs<sygraph::memory::space::shared>(q);
  buildAndDestroyGraphs<sygraph::memory::space::device>(q);
#endif
}
