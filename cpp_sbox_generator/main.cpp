#include "HybridReversibleSboxBuilder.h"
#include <iostream>

int main() {
    // Create the builder with a random seed and verbose output
    HybridReversibleSboxBuilder builder(42, true);
    
    // Set the gate set to use
    builder.set_gate_set(GateSetType::AGGRESSIVE);
    
    // Run the improved implementation with evolutionary optimization
    auto result = builder.implement_aes_sbox_improved(200, 20000, 0.15);
    
    // Print results
    QuantumCircuit best_circuit = result.first;
    std::cout << "Best circuit size: " << best_circuit.size() << " gates" << std::endl;
    
    // Verify the S-box implementation
    int correct_count = 0;
    auto& test_results = result.second;
    
    for (int i = 0; i < 256; ++i) {
        if (test_results[i]["correct"] == 1) {
            correct_count++;
        }
    }
    
    std::cout << "Correct S-box mappings: " << correct_count << "/256 (" 
              << (100.0 * correct_count / 256.0) << "%)" << std::endl;
    
    return 0;
}