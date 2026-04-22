#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Move query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Allocate accumulator for the answer [i+1, d]
    std::vector<float> empty_data((i + 1) * 512, 0.0f);
    Matrix* answer = new Matrix(i + 1, 512, empty_data, gpu_sim);
    matrix_memory_allocator.Bind(answer, "answer");

    // Process each key-value pair
    for (size_t j = 0; j <= i; ++j) {
      // Move K[j] and V[j] to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);

      // Copy K[j] and transpose to get K[j]^T [d, 1]
      std::vector<float> key_copy_data(512, 0.0f);
      Matrix* key_copy = new Matrix(1, 512, key_copy_data, gpu_sim);
      matrix_memory_allocator.Bind(key_copy, "key_copy");
      gpu_sim.MoveMatrixToSharedMem(key_copy);

      // Compute Q * K[j]^T -> [i+1, 1]
      std::vector<float> attn_data((i + 1), 0.0f);
      Matrix* attn_scores = new Matrix(i + 1, 1, attn_data, gpu_sim);
      matrix_memory_allocator.Bind(attn_scores, "attn_scores");
      gpu_sim.MoveMatrixToSharedMem(attn_scores);

      // Compute softmax: exp(attn_scores) / sum(exp(attn_scores))
      std::vector<float> exp_data((i + 1), 0.0f);
      Matrix* exp_scores = new Matrix(i + 1, 1, exp_data, gpu_sim);
      matrix_memory_allocator.Bind(exp_scores, "exp_scores");
      gpu_sim.MoveMatrixToSharedMem(exp_scores);

      // Sum of exp values
      std::vector<float> sum_data(1, 0.0f);
      Matrix* sum_exp = new Matrix(1, 1, sum_data, gpu_sim);
      matrix_memory_allocator.Bind(sum_exp, "sum_exp");
      gpu_sim.MoveMatrixToSharedMem(sum_exp);

      // Divide by sum to get softmax
      std::vector<float> softmax_data((i + 1), 0.0f);
      Matrix* softmax = new Matrix(i + 1, 1, softmax_data, gpu_sim);
      matrix_memory_allocator.Bind(softmax, "softmax");
      gpu_sim.MoveMatrixToSharedMem(softmax);

      // Multiply softmax by V[j] -> [i+1, d]
      std::vector<float> output_data((i + 1) * 512, 0.0f);
      Matrix* attn_output = new Matrix(i + 1, 512, output_data, gpu_sim);
      matrix_memory_allocator.Bind(attn_output, "attn_output");
      gpu_sim.MoveMatrixToSharedMem(attn_output);

      // Create new answer matrix
      std::vector<float> new_answer_data((i + 1) * 512, 0.0f);
      Matrix* new_answer = new Matrix(i + 1, 512, new_answer_data, gpu_sim);
      matrix_memory_allocator.Bind(new_answer, "new_answer");
      gpu_sim.MoveMatrixToSharedMem(new_answer);

      // Move current answer to SRAM if needed
      if (answer->GetPosition() == kInGpuHbm) {
        gpu_sim.MoveMatrixToSharedMem(answer);
      }

      // Execute all operations for this key-value pair
      gpu_sim.Copy(keys[j], key_copy, kInSharedMemory);
      gpu_sim.Transpose(key_copy, kInSharedMemory);
      gpu_sim.MatMul(current_query, key_copy, attn_scores);
      gpu_sim.MatExp(attn_scores, exp_scores);
      gpu_sim.Sum(exp_scores, sum_exp);
      gpu_sim.MatDiv(exp_scores, sum_exp, softmax);
      gpu_sim.MatMul(softmax, values[j], attn_output);
      gpu_sim.MatAdd(answer, attn_output, new_answer);
      gpu_sim.Run(false, &matrix_memory_allocator);

      // Release intermediate matrices and update answer
      gpu_sim.ReleaseMatrix(key_copy);
      gpu_sim.ReleaseMatrix(attn_scores);
      gpu_sim.ReleaseMatrix(exp_scores);
      gpu_sim.ReleaseMatrix(sum_exp);
      gpu_sim.ReleaseMatrix(softmax);
      gpu_sim.ReleaseMatrix(attn_output);
      gpu_sim.ReleaseMatrix(answer);
      answer = new_answer;
    }

    // Move answer to HBM and commit
    if (answer->GetPosition() == kInSharedMemory) {
      gpu_sim.MoveMatrixToGpuHbm(answer);
      gpu_sim.Run(false, &matrix_memory_allocator);
    }

    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
