#include "simulator.hpp"
#include <iostream>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  
  // Create matrices like in the problem
  std::vector<float> q_data(512, 1.0f);
  sjtu::Matrix* query = new sjtu::Matrix(1, 512, q_data, gpu_sim);
  matrix_memory_allocator.Bind(query, "query");
  
  std::vector<float> k_data(512, 2.0f);
  sjtu::Matrix* key = new sjtu::Matrix(1, 512, k_data, gpu_sim);
  matrix_memory_allocator.Bind(key, "key");
  
  std::vector<float> v_data(512, 3.0f);
  sjtu::Matrix* value = new sjtu::Matrix(1, 512, v_data, gpu_sim);
  matrix_memory_allocator.Bind(value, "value");
  
  std::cout << "Initial positions:" << std::endl;
  std::cout << "  query: " << (int)query->GetPosition() << std::endl;
  std::cout << "  key: " << (int)key->GetPosition() << std::endl;
  std::cout << "  value: " << (int)value->GetPosition() << std::endl;
  
  // Move to SRAM
  gpu_sim.MoveMatrixToSharedMem(query);
  gpu_sim.MoveMatrixToSharedMem(key);
  gpu_sim.MoveMatrixToSharedMem(value);
  
  // Copy and transpose key
  std::vector<float> key_copy_data(512, 0.0f);
  sjtu::Matrix* key_copy = new sjtu::Matrix(1, 512, key_copy_data, gpu_sim);
  matrix_memory_allocator.Bind(key_copy, "key_copy");
  gpu_sim.MoveMatrixToSharedMem(key_copy);
  gpu_sim.Copy(key, key_copy, sjtu::kInSharedMemory);
  gpu_sim.Transpose(key_copy, sjtu::kInSharedMemory);
  
  // MatMul: Q * K^T
  std::vector<float> attn_data(1, 0.0f);
  sjtu::Matrix* attn_scores = new sjtu::Matrix(1, 1, attn_data, gpu_sim);
  matrix_memory_allocator.Bind(attn_scores, "attn_scores");
  gpu_sim.MoveMatrixToSharedMem(attn_scores);
  gpu_sim.MatMul(query, key_copy, attn_scores);
  
  std::cout << "\nRunning simulator..." << std::endl;
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  std::cout << "\nAfter Run:" << std::endl;
  std::cout << "  query: " << (int)query->GetPosition() << std::endl;
  std::cout << "  key: " << (int)key->GetPosition() << std::endl;
  std::cout << "  key_copy: " << (int)key_copy->GetPosition() << std::endl;
  std::cout << "  attn_scores: " << (int)attn_scores->GetPosition() << std::endl;
  
  return 0;
}
