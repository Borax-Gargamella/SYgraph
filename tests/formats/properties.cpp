#include <cassert>
#include <sstream>

#include <sygraph/sygraph.hpp>

int main() {
  using namespace sygraph;

  // Matrix Market symmetric pattern: should be undirected and unweighted
  std::string symmetric_mm = "%%MatrixMarket matrix coordinate pattern symmetric\n"
                             "%\n"
                             "3 3 2\n"
                             "1 2\n"
                             "2 3\n";
  std::istringstream symmetric_stream(symmetric_mm);
  graph::Properties symmetric_properties;
  auto symmetric_csr = io::csr::fromMM<uint, uint, uint>(symmetric_stream, &symmetric_properties);
  assert(symmetric_csr.getNumNonzeros() == 4); // symmetric entries duplicated
  assert(!symmetric_properties.directed);
  assert(!symmetric_properties.weighted);

  // Matrix Market general real: directed and weighted
  std::string general_mm = "%%MatrixMarket matrix coordinate real general\n"
                           "%\n"
                           "3 3 2\n"
                           "1 2 1.5\n"
                           "2 3 2.5\n";
  std::istringstream general_stream(general_mm);
  graph::Properties general_properties;
  auto general_csr = io::csr::fromMM<float, uint, uint>(general_stream, &general_properties);
  assert(general_csr.getNumNonzeros() == 2);
  assert(general_properties.directed);
  assert(general_properties.weighted);

  // COO parsing should mark undirected graphs and detect explicit weights
  std::string coo_text = "% comment\n3 3 2\n0 1\n1 2 3\n";
  std::istringstream coo_stream(coo_text);
  graph::Properties coo_properties;
  auto coo = io::coo::fromCOO<float, uint, uint>(coo_stream, true, &coo_properties);
  assert(coo_properties.weighted);
  assert(!coo_properties.directed);
  auto csr_from_coo = io::csr::fromCOO(coo);
  assert(csr_from_coo.getRowOffsetsSize() == 3);

  // Binary serialization should round trip properties
  std::stringstream binary_stream(std::ios::in | std::ios::out | std::ios::binary);
  io::csr::toBinary(general_csr, binary_stream, general_properties);
  binary_stream.seekg(0, std::ios::beg);
  graph::Properties binary_properties;
  auto from_binary = io::csr::fromBinary<float, uint, uint>(binary_stream, &binary_properties);
  assert(binary_properties.directed == general_properties.directed);
  assert(binary_properties.weighted == general_properties.weighted);
  assert(from_binary.getNumNonzeros() == general_csr.getNumNonzeros());

  return 0;
}
