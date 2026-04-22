#include "simulator.hpp"
#include <iostream>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  
  // Test creating a matrix
  std::vector<float> data(10, 1.0f);
  sjtu::Matrix* m1 = new sjtu::Matrix(2, 5, data, gpu_sim);
  matrix_memory_allocator.Bind(m1, "m1");
  
  std::cout << "Matrix position: " << (int)m1->GetPosition() << std::endl;
  std::cout << "Matrix size: " << m1->GetSize() << std::endl;
  
  // Test moving to SRAM
  gpu_sim.MoveMatrixToSharedMem(m1);
  
  // Test a calculation
  sjtu::Matrix* m2 = new sjtu::Matrix(2, 5, data, gpu_sim);
  matrix_memory_allocator.Bind(m2, "m2");
  gpu_sim.MoveMatrixToSharedMem(m2);
  
  sjtu::Matrix* result = new sjtu::Matrix(2, 5, data, gpu_sim);
  matrix_memory_allocator.Bind(result, "result");
  gpu_sim.MoveMatrixToSharedMem(result);
  
  gpu_sim.MatAdd(m1, m2, result);
  
  std::cout << "Running simulator..." << std::endl;
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  std::cout << "Done" << std::endl;
  return 0;
}
