#include "simulator.hpp"
#include <iostream>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  
  // Create matrices
  std::vector<float> q_data(512, 1.0f);
  sjtu::Matrix* query = new sjtu::Matrix(1, 512, q_data, gpu_sim);
  matrix_memory_allocator.Bind(query, "query");
  
  std::vector<float> k_data(512, 2.0f);
  sjtu::Matrix* key = new sjtu::Matrix(1, 512, k_data, gpu_sim);
  matrix_memory_allocator.Bind(key, "key");
  
  std::vector<float> v_data(512, 3.0f);
  sjtu::Matrix* value = new sjtu::Matrix(1, 512, v_data, gpu_sim);
  matrix_memory_allocator.Bind(value, "value");
  
  // Queue all instructions
  gpu_sim.MoveMatrixToSharedMem(query);
  gpu_sim.MoveMatrixToSharedMem(key);
  gpu_sim.MoveMatrixToSharedMem(value);
  
  std::vector<float> key_copy_data(512, 0.0f);
  sjtu::Matrix* key_copy = new sjtu::Matrix(1, 512, key_copy_data, gpu_sim);
  matrix_memory_allocator.Bind(key_copy, "key_copy");
  gpu_sim.MoveMatrixToSharedMem(key_copy);
  gpu_sim.Copy(key, key_copy, sjtu::kInSharedMemory);
  gpu_sim.Transpose(key_copy, sjtu::kInSharedMemory);
  
  std::vector<float> attn_data(1, 0.0f);
  sjtu::Matrix* attn_scores = new sjtu::Matrix(1, 1, attn_data, gpu_sim);
  matrix_memory_allocator.Bind(attn_scores, "attn_scores");
  gpu_sim.MoveMatrixToSharedMem(attn_scores);
  gpu_sim.MatMul(query, key_copy, attn_scores);
  
  std::vector<float> exp_data(1, 0.0f);
  sjtu::Matrix* exp_scores = new sjtu::Matrix(1, 1, exp_data, gpu_sim);
  matrix_memory_allocator.Bind(exp_scores, "exp_scores");
  gpu_sim.MoveMatrixToSharedMem(exp_scores);
  gpu_sim.MatExp(attn_scores, exp_scores);
  
  std::vector<float> sum_data(1, 0.0f);
  sjtu::Matrix* sum_exp = new sjtu::Matrix(1, 1, sum_data, gpu_sim);
  matrix_memory_allocator.Bind(sum_exp, "sum_exp");
  gpu_sim.MoveMatrixToSharedMem(sum_exp);
  gpu_sim.Sum(exp_scores, sum_exp);
  
  std::vector<float> softmax_data(1, 0.0f);
  sjtu::Matrix* softmax = new sjtu::Matrix(1, 1, softmax_data, gpu_sim);
  matrix_memory_allocator.Bind(softmax, "softmax");
  gpu_sim.MoveMatrixToSharedMem(softmax);
  gpu_sim.MatDiv(exp_scores, sum_exp, softmax);
  
  std::vector<float> output_data(512, 0.0f);
  sjtu::Matrix* attn_output = new sjtu::Matrix(1, 512, output_data, gpu_sim);
  matrix_memory_allocator.Bind(attn_output, "attn_output");
  gpu_sim.MoveMatrixToSharedMem(attn_output);
  gpu_sim.MatMul(softmax, value, attn_output);
  
  std::vector<float> answer_data(512, 0.0f);
  sjtu::Matrix* answer = new sjtu::Matrix(1, 512, answer_data, gpu_sim);
  matrix_memory_allocator.Bind(answer, "answer");
  gpu_sim.MoveMatrixToSharedMem(answer);
  
  std::vector<float> new_answer_data(512, 0.0f);
  sjtu::Matrix* new_answer = new sjtu::Matrix(1, 512, new_answer_data, gpu_sim);
  matrix_memory_allocator.Bind(new_answer, "new_answer");
  gpu_sim.MoveMatrixToSharedMem(new_answer);
  gpu_sim.MatAdd(answer, attn_output, new_answer);
  
  gpu_sim.ReleaseMatrix(key_copy);
  gpu_sim.ReleaseMatrix(attn_scores);
  gpu_sim.ReleaseMatrix(exp_scores);
  gpu_sim.ReleaseMatrix(sum_exp);
  gpu_sim.ReleaseMatrix(softmax);
  gpu_sim.ReleaseMatrix(attn_output);
  gpu_sim.ReleaseMatrix(answer);
  
  gpu_sim.MoveMatrixToGpuHbm(key);
  gpu_sim.MoveMatrixToGpuHbm(value);
  gpu_sim.MoveMatrixToGpuHbm(new_answer);
  
  std::cout << "Running simulator..." << std::endl;
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  std::cout << "Done! new_answer position: " << (int)new_answer->GetPosition() << std::endl;
  
  return 0;
}
