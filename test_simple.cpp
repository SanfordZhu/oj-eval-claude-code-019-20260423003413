#include "simulator.hpp"
#include <iostream>
#include <cmath>

int main() {
  sjtu::GpuSimulator gpu_sim;
  sjtu::MatrixMemoryAllocator matrix_memory_allocator;
  
  // Simple test: Q = [2, 1], K = [1, 1], V = [1, 1]
  // Q * K^T = [2, 1] * [1, 1] = [2, 1]
  // softmax([2, 1]) = [2, 1] (each element normalized by sum of exp)
  // softmax * V = [2, 1] * [1, 1] = [2, 1]
  
  std::vector<float> q_data = {1.0f, 2.0f};
  sjtu::Matrix* query = new sjtu::Matrix(2, 1, q_data, gpu_sim);
  matrix_memory_allocator.Bind(query, "query");
  
  std::vector<float> k_data = {3.0f};
  sjtu::Matrix* key = new sjtu::Matrix(1, 1, k_data, gpu_sim);
  matrix_memory_allocator.Bind(key, "key");
  
  std::vector<float> v_data = {4.0f};
  sjtu::Matrix* value = new sjtu::Matrix(1, 1, v_data, gpu_sim);
  matrix_memory_allocator.Bind(value, "value");
  
  gpu_sim.MoveMatrixToSharedMem(query);
  gpu_sim.MoveMatrixToSharedMem(key);
  gpu_sim.MoveMatrixToSharedMem(value);
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  // Transpose key
  gpu_sim.Transpose(key, sjtu::kInSharedMemory);
  
  // Q * K^T
  std::vector<float> attn_data(2, 0.0f);
  sjtu::Matrix* attn_scores = new sjtu::Matrix(2, 1, attn_data, gpu_sim);
  matrix_memory_allocator.Bind(attn_scores, "attn_scores");
  gpu_sim.MoveMatrixToSharedMem(attn_scores);
  gpu_sim.MatMul(query, key, attn_scores);
  
  // Exp
  std::vector<float> exp_data(2, 0.0f);
  sjtu::Matrix* exp_scores = new sjtu::Matrix(2, 1, exp_data, gpu_sim);
  matrix_memory_allocator.Bind(exp_scores, "exp_scores");
  gpu_sim.MoveMatrixToSharedMem(exp_scores);
  gpu_sim.MatExp(attn_scores, exp_scores);
  
  // Sum
  std::vector<float> sum_data(1, 0.0f);
  sjtu::Matrix* sum_exp = new sjtu::Matrix(1, 1, sum_data, gpu_sim);
  matrix_memory_allocator.Bind(sum_exp, "sum_exp");
  gpu_sim.MoveMatrixToSharedMem(sum_exp);
  gpu_sim.Sum(exp_scores, sum_exp);
  
  // Div
  std::vector<float> softmax_data(2, 0.0f);
  sjtu::Matrix* softmax = new sjtu::Matrix(2, 1, softmax_data, gpu_sim);
  matrix_memory_allocator.Bind(softmax, "softmax");
  gpu_sim.MoveMatrixToSharedMem(softmax);
  gpu_sim.MatDiv(exp_scores, sum_exp, softmax);
  
  // Softmax * V
  std::vector<float> output_data(2, 0.0f);
  sjtu::Matrix* attn_output = new sjtu::Matrix(2, 1, output_data, gpu_sim);
  matrix_memory_allocator.Bind(attn_output, "attn_output");
  gpu_sim.MoveMatrixToSharedMem(attn_output);
  gpu_sim.MatMul(softmax, value, attn_output);
  
  gpu_sim.Run(false, &matrix_memory_allocator);
  
  std::cout << "Q: ";
  for (size_t i = 0; i < query->GetSize(); ++i) {
    std::cout << query->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "K: ";
  for (size_t i = 0; i < key->GetSize(); ++i) {
    std::cout << key->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "V: ";
  for (size_t i = 0; i < value->GetSize(); ++i) {
    std::cout << value->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Q * K^T: ";
  for (size_t i = 0; i < attn_scores->GetSize(); ++i) {
    std::cout << attn_scores->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Exp: ";
  for (size_t i = 0; i < exp_scores->GetSize(); ++i) {
    std::cout << exp_scores->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Sum: " << sum_exp->data_[0] << std::endl;
  
  std::cout << "Softmax: ";
  for (size_t i = 0; i < softmax->GetSize(); ++i) {
    std::cout << softmax->data_[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Output: ";
  for (size_t i = 0; i < attn_output->GetSize(); ++i) {
    std::cout << attn_output->data_[i] << " ";
  }
  std::cout << std::endl;
  
  // Manual calculation
  float q0 = 1.0f, q1 = 2.0f, k = 3.0f, v = 4.0f;
  float attn0 = q0 * k;  // 1 * 3 = 3
  float attn1 = q1 * k;  // 2 * 3 = 6
  float exp0 = exp(attn0);  // exp(3) = 20.0855
  float exp1 = exp(attn1);  // exp(6) = 403.4288
  float sum_exp = exp0 + exp1;  // 423.5143
  float softmax0 = exp0 / sum_exp;  // 20.0855 / 423.5143 = 0.0474
  float softmax1 = exp1 / sum_exp;  // 403.4288 / 423.5143 = 0.9526
  float output0 = softmax0 * v;  // 0.0474 * 4 = 0.1896
  float output1 = softmax1 * v;  // 0.9526 * 4 = 3.8104
  
  std::cout << "Expected output: " << output0 << " " << output1 << std::endl;
  
  return 0;
}
