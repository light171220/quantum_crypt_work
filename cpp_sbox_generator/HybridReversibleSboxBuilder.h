#ifndef HYBRID_REVERSIBLE_SBOX_BUILDER_H
#define HYBRID_REVERSIBLE_SBOX_BUILDER_H

#include "QuantumCircuit.h"
#include "QuantumSimulator.h"
#include "Constants.h"
#include <random>
#include <vector>
#include <unordered_map>
#include <functional>

enum class GateSetType {
    BASIC,
    EXTENDED,
    AGGRESSIVE,
    CUSTOM
};

class HybridReversibleSboxBuilder {
public:
    HybridReversibleSboxBuilder(unsigned int seed = 42, bool verbose = false);
    
    void set_gate_set(GateSetType gate_set);
    void set_verbose(bool verbose);
    
    std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
    implement_aes_sbox(int population_size, int generations, double mutation_rate);
    
    std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
    implement_aes_sbox_improved(int population_size, int generations, double mutation_rate);
    
private:
    std::vector<GateDefinition> basic_gate_set();
    std::vector<GateDefinition> extended_gate_set();
    std::vector<GateDefinition> aggressive_gate_set();
    std::vector<GateDefinition> custom_gate_set();
    std::vector<GateDefinition> build_gate_set();
    
    double calculate_population_diversity(const std::vector<QuantumCircuit>& population);
    
    QuantumCircuit create_field_inversion_circuit();
    QuantumCircuit create_optimized_field_inversion_circuit();
    QuantumCircuit create_affine_transform_circuit();
    QuantumCircuit create_optimized_affine_transform_circuit();
    QuantumCircuit create_complete_aes_sbox_circuit();
    QuantumCircuit direct_aes_sbox_construction();
    
    std::vector<QuantumCircuit> generate_initial_population(int population_size, int circuit_size);
    std::vector<QuantumCircuit> generate_field_inversion_population(int population_size);
    std::vector<QuantumCircuit> generate_sbox_population_with_fixed_inversion(int population_size, const QuantumCircuit& inversion_circuit);
    
    double evaluate_circuit(const QuantumCircuit& circuit);
    double improved_evaluate_circuit(const QuantumCircuit& circuit);
    double evaluate_field_inversion(const QuantumCircuit& circuit);
    
    QuantumCircuit crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2);
    QuantumCircuit crossover_with_diversity(const QuantumCircuit& parent1, const QuantumCircuit& parent2, double diversity_score);
    QuantumCircuit multi_point_crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2, int num_points);
    QuantumCircuit uniform_crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2);
    
    QuantumCircuit adaptive_mutate(const QuantumCircuit& circuit, double base_mutation_rate, double diversity_score, int generation);
    QuantumCircuit improved_adaptive_mutate(const QuantumCircuit& circuit, double base_mutation_rate, double correctness_score, int generation);
    
    int tournament_selection(const std::vector<double>& scores, double diversity_score);
    
    QuantumCircuit simulated_annealing_optimize(QuantumCircuit circuit, int iterations, double initial_temp);
    
    std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
    implement_aes_sbox_hybrid(int population_size, int generations, double mutation_rate);
    
    double calculate_entropy(const std::unordered_map<uint64_t, uint64_t>& mapping);
    double calculate_avalanche(const std::unordered_map<uint64_t, uint64_t>& mapping, int num_bits);
    
    std::mt19937 rng;
    QuantumSimulator simulator;
    
    int quantum_cost;
    int gate_count;
    int two_qubit_cost;
    
    bool verbose;
    GateSetType current_gate_set;
};

#endif // HYBRID_REVERSIBLE_SBOX_BUILDER_H