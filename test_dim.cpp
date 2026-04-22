#include "simulator.hpp"
#include <iostream>

int main() {
  // Check dimensions of matrices from main.cpp
  // keys[i] and values[i] are [1, 512]
  // queries[i] are [i+1, 512] - so for i=0, Q is [1, 512]; for i=1, Q is [2, 512], etc.
  // answers[i] are [i+1, 512]
  
  std::cout << "Round 0: Q is [1, 512], K[0] is [1, 512], V[0] is [1, 512]" << std::endl;
  std::cout << "  Q * K[0]^T: [1, 512] * [512, 1] = [1, 1]" << std::endl;
  std::cout << "  softmax([1, 1]): [1, 1]" << std::endl;
  std::cout << "  softmax * V[0]: [1, 1] * [1, 512] = [1, 512]" << std::endl;
  std::cout << "  Answer[0]: [1, 512]" << std::endl;
  
  std::cout << "\nRound 1: Q is [2, 512], K[0],K[1] are [1, 512], V[0],V[1] are [1, 512]" << std::endl;
  std::cout << "  For j=0: Q * K[0]^T: [2, 512] * [512, 1] = [2, 1]" << std::endl;
  std::cout << "  softmax([2, 1]): [2, 1]" << std::endl;
  std::cout << "  softmax * V[0]: [2, 1] * [1, 512] = [2, 512]" << std::endl;
  std::cout << "  For j=1: Q * K[1]^T: [2, 512] * [512, 1] = [2, 1]" << std::endl;
  std::cout << "  softmax([2, 1]): [2, 1]" << std::endl;
  std::cout << "  softmax * V[1]: [2, 1] * [1, 512] = [2, 512]" << std::endl;
  std::cout << "  Sum: [2, 512] + [2, 512] = [2, 512]" << std::endl;
  std::cout << "  Answer[1]: [2, 512]" << std::endl;
  
  return 0;
}
