#include "simulator.hpp"
#include <iostream>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  
  // Test creating a matrix like keys
  std::vector<float> data(512, 1.0f);
  sjtu::Matrix* key = new sjtu::Matrix(1, 512, data, gpu_sim);
  matrix_memory_allocator.Bind(key, "key");
  
  std::cout << "Key position: " << (int)key->GetPosition() << " (0=HBM, 1=SRAM, 2=Released)" << std::endl;
  
  // Now try to move to SRAM
  std::cout << "Moving to SRAM..." << std::endl;
  gpu_sim.MoveMatrixToSharedMem(key);
  
  // Check position before running
  std::cout << "Key position after MoveMatrixToSharedMem: " << (int)key->GetPosition() << std::endl;
  
  // Run the simulator
  std::cout << "Running simulator..." << std::endl;
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  std::cout << "Key position after Run: " << (int)key->GetPosition() << std::endl;
  return 0;
}
