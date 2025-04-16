#include "HybridReversibleSboxBuilder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>
#include <thread>
#include <mutex>
#include <random>
#include <set>

HybridReversibleSboxBuilder::HybridReversibleSboxBuilder(unsigned int seed, bool verbose) 
    : rng(seed), quantum_cost(0), gate_count(0), two_qubit_cost(0), 
      simulator(seed), verbose(verbose), current_gate_set(GateSetType::AGGRESSIVE) {}

void HybridReversibleSboxBuilder::set_gate_set(GateSetType gate_set) {
    current_gate_set = gate_set;
}

void HybridReversibleSboxBuilder::set_verbose(bool v) {
    verbose = v;
}

std::vector<GateDefinition> HybridReversibleSboxBuilder::basic_gate_set() {
    return {
        {"NOT", 1, 0, 1, 0, 1, 
            [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
                circ.x(qubits[0]); 
            }
        },
        {"CNOT", 2, 1, 1, 1, 1, 
            [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
                circ.cx(qubits[0], qubits[1]); 
            }
        },
        {"Toffoli", 3, 2, 5, 5, 1, 
            [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
                circ.ccx(qubits[0], qubits[1], qubits[2]); 
            }
        }
    };
}

std::vector<GateDefinition> HybridReversibleSboxBuilder::extended_gate_set() {
    auto gates = basic_gate_set();
    gates.push_back({"SWAP", 2, 0, 3, 3, 1, 
        [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
            circ.swap(qubits[0], qubits[1]); 
        }
    });
    return gates;
}

std::vector<GateDefinition> HybridReversibleSboxBuilder::aggressive_gate_set() {
    auto gates = extended_gate_set();
    gates.push_back({"Fredkin", 3, 1, 7, 7, 1, 
        [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
            circ.fredkin(qubits[0], qubits[1], qubits[2]); 
        }
    });
    gates.push_back({"MCX-3", 4, 3, 12, 12, 1, 
        [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
            std::vector<int> controls = {qubits[0], qubits[1], qubits[2]};
            circ.multi_controlled_x(controls, qubits[3]);
        }
    });
    gates.push_back({"MCX-4", 5, 4, 16, 16, 1, 
        [](QuantumCircuit& circ, const std::vector<int>& qubits) { 
            std::vector<int> controls = {qubits[0], qubits[1], qubits[2], qubits[3]};
            circ.multi_controlled_x(controls, qubits[4]);
        }
    });
    return gates;
}

std::vector<GateDefinition> HybridReversibleSboxBuilder::custom_gate_set() {
    return aggressive_gate_set();
}

std::vector<GateDefinition> HybridReversibleSboxBuilder::build_gate_set() {
    switch (current_gate_set) {
        case GateSetType::BASIC:
            return basic_gate_set();
        case GateSetType::EXTENDED:
            return extended_gate_set();
        case GateSetType::AGGRESSIVE:
            return aggressive_gate_set();
        case GateSetType::CUSTOM:
            return custom_gate_set();
        default:
            return basic_gate_set();
    }
}

double HybridReversibleSboxBuilder::calculate_population_diversity(
    const std::vector<QuantumCircuit>& population) {
    std::set<std::string> unique_circuits;
    for (const auto& circuit : population) {
        std::string fingerprint = "";
        for (const auto& gate : circuit.gates) {
            fingerprint += std::to_string(static_cast<int>(gate.type)) + ":";
            for (int qubit : gate.qubits) {
                fingerprint += std::to_string(qubit) + ",";
            }
            fingerprint += ";";
        }
        unique_circuits.insert(fingerprint);
    }
    return static_cast<double>(unique_circuits.size()) / population.size();
}

QuantumCircuit HybridReversibleSboxBuilder::create_field_inversion_circuit() {
    QuantumCircuit circuit(8, 8);
    for (int i = 0; i < 4; i++) {
        circuit.swap(i, 7-i);
    }
    for (int i = 0; i < 7; i++) {
        circuit.cx(i, i+1);
    }
    for (int i = 0; i < 3; i++) {
        circuit.ccx(i, i+1, i+2);
    }
    return circuit;
}

QuantumCircuit HybridReversibleSboxBuilder::create_optimized_field_inversion_circuit() {
    QuantumCircuit circuit(8, 8);
    
    circuit.x(7);
    circuit.cx(7, 6);
    circuit.cx(6, 5);
    circuit.cx(5, 4);
    circuit.cx(4, 3);
    circuit.cx(4, 1);
    
    circuit.cx(7, 2);
    circuit.cx(5, 7);
    circuit.cx(3, 6);
    circuit.cx(5, 3);
    circuit.cx(1, 5);
    circuit.cx(7, 1);
    circuit.cx(4, 0);
    circuit.cx(1, 4);
    
    circuit.cx(7, 3);
    circuit.cx(6, 2);
    circuit.cx(5, 1);
    circuit.cx(4, 0);
    
    for (int i = 0; i < 3; i++) {
        circuit.ccx(i, i+1, i+2);
    }
    
    circuit.cx(1, 5);
    circuit.cx(5, 3);
    circuit.cx(3, 7);
    circuit.cx(7, 6);
    circuit.cx(6, 5);
    circuit.cx(0, 4);
    
    return circuit;
}

QuantumCircuit HybridReversibleSboxBuilder::create_affine_transform_circuit() {
    QuantumCircuit circuit(8, 8);
    for (int i = 0; i < 8; i++) {
        circuit.cx((i+4) % 8, i);
        circuit.cx((i+5) % 8, i);
        circuit.cx((i+6) % 8, i);
        circuit.cx((i+7) % 8, i);
    }
    circuit.x(0);
    circuit.x(1);
    circuit.x(6);
    return circuit;
}

QuantumCircuit HybridReversibleSboxBuilder::create_optimized_affine_transform_circuit() {
    QuantumCircuit circuit(8, 8);
    
    for (int i = 0; i < 8; i++) {
        circuit.cx((i+4) % 8, i);
        circuit.cx((i+5) % 8, i);
        circuit.cx((i+6) % 8, i);
        circuit.cx((i+7) % 8, i);
    }
    
    circuit.x(0);
    circuit.x(1);
    circuit.x(6);
    
    return circuit;
}

QuantumCircuit HybridReversibleSboxBuilder::create_complete_aes_sbox_circuit() {
    QuantumCircuit circuit(8, 8);
    
    QuantumCircuit inversion = create_optimized_field_inversion_circuit();
    circuit.append(inversion);
    
    QuantumCircuit affine = create_optimized_affine_transform_circuit();
    circuit.append(affine);
    
    return circuit;
}

std::vector<QuantumCircuit> HybridReversibleSboxBuilder::generate_initial_population(
    int population_size, int circuit_size) {
    std::vector<QuantumCircuit> population;
    auto gates = build_gate_set();
    std::uniform_int_distribution<> gate_dist(0, gates.size() - 1);
    
    for (int p = 0; p < population_size; ++p) {
        QuantumCircuit circuit(8, 8);
        
        if (p < population_size * 0.25) {
            QuantumCircuit inversion = create_optimized_field_inversion_circuit();
            QuantumCircuit affine = create_optimized_affine_transform_circuit();
            
            circuit.append(inversion);
            circuit.append(affine);
            
            int additional_gates = std::uniform_int_distribution<>(5, 15)(rng);
            for (int i = 0; i < additional_gates; ++i) {
                auto gate = gates[gate_dist(rng)];
                std::vector<int> qubits(gate.qubits);
                for (int j = 0; j < gate.qubits; ++j) {
                    qubits[j] = std::uniform_int_distribution<>(0, 7)(rng);
                }
                
                std::sort(qubits.begin(), qubits.end());
                qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
                
                if (qubits.size() >= static_cast<size_t>(gate.qubits)) {
                    gate.apply(circuit, std::vector<int>(qubits.begin(), qubits.begin() + gate.qubits));
                }
            }
        } 
        else if (p < population_size * 0.5) {
            for (int i = 0; i < circuit_size / 3; ++i) {
                int pattern_type = std::uniform_int_distribution<>(0, 2)(rng);
                
                if (pattern_type == 0) {
                    for (int j = 0; j < 7; ++j) {
                        circuit.cx(j, j+1);
                    }
                }
                else if (pattern_type == 1) {
                    for (int j = 0; j < 6; ++j) {
                        circuit.ccx(j, j+1, j+2);
                    }
                }
                else {
                    for (int j = 0; j < 4; ++j) {
                        circuit.swap(j, 7-j);
                    }
                }
            }
        }
        else {
            int num_gates = std::uniform_int_distribution<>(circuit_size/2, circuit_size)(rng);
            
            for (int g = 0; g < num_gates; ++g) {
                auto gate = gates[gate_dist(rng)];
                std::vector<int> available_qubits = {0, 1, 2, 3, 4, 5, 6, 7};
                std::shuffle(available_qubits.begin(), available_qubits.end(), rng);
                
                if (available_qubits.size() >= static_cast<size_t>(gate.qubits)) {
                    gate.apply(circuit, std::vector<int>(available_qubits.begin(), 
                                                     available_qubits.begin() + gate.qubits));
                }
            }
        }
        population.push_back(circuit);
    }
    return population;
}

std::vector<QuantumCircuit> HybridReversibleSboxBuilder::generate_field_inversion_population(int population_size) {
    std::vector<QuantumCircuit> population;
    
    for (int i = 0; i < population_size; i++) {
        if (i < population_size * 0.2) {
            population.push_back(create_optimized_field_inversion_circuit());
        } else if (i < population_size * 0.4) {
            QuantumCircuit circuit = create_optimized_field_inversion_circuit();
            
            std::uniform_int_distribution<> num_mutations(1, 5);
            int mutations = num_mutations(rng);
            
            for (int j = 0; j < mutations; j++) {
                std::uniform_int_distribution<> gate_idx_dist(0, circuit.gates.size() - 1);
                int idx = gate_idx_dist(rng);
                circuit.gates.erase(circuit.gates.begin() + idx);
            }
            
            population.push_back(circuit);
        } else {
            QuantumCircuit circuit(8, 8);
            
            std::uniform_int_distribution<> num_gates(15, 30);
            int gate_count = num_gates(rng);
            
            auto gates = build_gate_set();
            std::uniform_int_distribution<> gate_type_dist(0, gates.size() - 1);
            
            for (int j = 0; j < gate_count; j++) {
                auto gate = gates[gate_type_dist(rng)];
                
                std::vector<int> qubits(8);
                for (int k = 0; k < 8; k++) {
                    qubits[k] = k;
                }
                std::shuffle(qubits.begin(), qubits.end(), rng);
                
                if (qubits.size() >= static_cast<size_t>(gate.qubits)) {
                    gate.apply(circuit, std::vector<int>(qubits.begin(), qubits.begin() + gate.qubits));
                }
            }
            
            population.push_back(circuit);
        }
    }
    
    return population;
}

std::vector<QuantumCircuit> HybridReversibleSboxBuilder::generate_sbox_population_with_fixed_inversion(
    int population_size, const QuantumCircuit& inversion_circuit) {
    
    std::vector<QuantumCircuit> population;
    
    for (int i = 0; i < population_size; i++) {
        if (i < population_size * 0.3) {
            QuantumCircuit circuit(8, 8);
            circuit.append(inversion_circuit);
            circuit.append(create_optimized_affine_transform_circuit());
            population.push_back(circuit);
        } else if (i < population_size * 0.6) {
            QuantumCircuit circuit(8, 8);
            circuit.append(inversion_circuit);
            
            auto gates = build_gate_set();
            std::uniform_int_distribution<> gate_type_dist(0, gates.size() - 1);
            std::uniform_int_distribution<> num_gates(10, 20);
            int gate_count = num_gates(rng);
            
            for (int j = 0; j < gate_count; j++) {
                auto gate = gates[gate_type_dist(rng)];
                
                std::vector<int> qubits(8);
                for (int k = 0; k < 8; k++) {
                    qubits[k] = k;
                }
                std::shuffle(qubits.begin(), qubits.end(), rng);
                
                if (qubits.size() >= static_cast<size_t>(gate.qubits)) {
                    gate.apply(circuit, std::vector<int>(qubits.begin(), qubits.begin() + gate.qubits));
                }
            }
            
            population.push_back(circuit);
        } else {
            population.push_back(create_complete_aes_sbox_circuit());
        }
    }
    
    return population;
}

double HybridReversibleSboxBuilder::evaluate_circuit(const QuantumCircuit& circuit) {
    std::unordered_map<uint64_t, uint64_t> mapping;
    int correct_outputs = 0;
    int total_inputs = 256;
    
    for (int input = 0; input < total_inputs; ++input) {
        uint64_t output = simulator.simulate(circuit, input);
        mapping[input] = output;
        
        if (output == AES_SBOX[input]) {
            correct_outputs++;
        }
    }
    
    double correctness_score = static_cast<double>(correct_outputs) / total_inputs;
    double entropy = calculate_entropy(mapping);
    double avalanche = calculate_avalanche(mapping, 8);
    
    double circuit_size_penalty = 0.0;
    if (circuit.gates.size() > 300) {
        circuit_size_penalty = 0.1 * (circuit.gates.size() / 600.0);
    }
    
    double final_score = (
        0.8 * correctness_score +
        0.15 * (entropy / 8.0) +
        0.15 * avalanche -
        circuit_size_penalty
    );
    
    return std::max(0.0, std::min(1.0, final_score));
}

double HybridReversibleSboxBuilder::improved_evaluate_circuit(const QuantumCircuit& circuit) {
    std::unordered_map<uint64_t, uint64_t> mapping;
    int correct_outputs = 0;
    int total_inputs = 256;
    
    for (int input = 0; input < total_inputs; ++input) {
        uint64_t output = simulator.simulate(circuit, input);
        mapping[input] = output;
        
        if (output == AES_SBOX[input]) {
            correct_outputs++;
        }
    }
    
    double correctness_score = static_cast<double>(correct_outputs) / total_inputs;
    
    double entropy = calculate_entropy(mapping);
    double avalanche = calculate_avalanche(mapping, 8);
    
    double circuit_size_penalty = 0.0;
    if (circuit.gates.size() > 100) {
        circuit_size_penalty = std::min(0.1, 0.1 * (circuit.gates.size() - 100) / 300.0);
    }
    
    double entropy_weight, avalanche_weight;
    
    if (correctness_score < 0.3) {
        entropy_weight = 0.05;
        avalanche_weight = 0.05;
    } else if (correctness_score < 0.7) {
        entropy_weight = 0.1;
        avalanche_weight = 0.1;
    } else {
        entropy_weight = 0.15;
        avalanche_weight = 0.15;
    }
    
    double final_score = (
        (1.0 - entropy_weight - avalanche_weight) * correctness_score +
        entropy_weight * (entropy / 8.0) +
        avalanche_weight * avalanche -
        circuit_size_penalty
    );
    
    return std::max(0.0, std::min(1.0, final_score));
}

double HybridReversibleSboxBuilder::evaluate_field_inversion(const QuantumCircuit& circuit) {
    std::unordered_map<uint64_t, uint64_t> mapping;
    int correct_outputs = 0;
    int non_zero_outputs = 0;
    
    for (int input = 1; input < 256; ++input) {
        uint64_t output = simulator.simulate(circuit, input);
        mapping[input] = output;
        
        uint64_t expected = 0;
        for (int i = 1; i < 256; i++) {
            if ((input * i) % 255 == 1) {
                expected = i;
                break;
            }
        }
        
        if (output == expected) {
            correct_outputs++;
        }
        
        if (output != 0) {
            non_zero_outputs++;
        }
    }
    
    double correctness_score = static_cast<double>(correct_outputs) / 255.0;
    double non_zero_score = static_cast<double>(non_zero_outputs) / 255.0;
    
    double circuit_size_penalty = 0.0;
    if (circuit.gates.size() > 50) {
        circuit_size_penalty = std::min(0.1, 0.1 * (circuit.gates.size() - 50) / 100.0);
    }
    
    double final_score = 0.8 * correctness_score + 0.2 * non_zero_score - circuit_size_penalty;
    
    return std::max(0.0, std::min(1.0, final_score));
}

QuantumCircuit HybridReversibleSboxBuilder::crossover_with_diversity(
    const QuantumCircuit& parent1, 
    const QuantumCircuit& parent2,
    double diversity_score) {
    
    QuantumCircuit child = parent1.copy();
    child.gates.clear();
    
    if (diversity_score < 0.3) {
        return uniform_crossover(parent1, parent2);
    }
    else if (diversity_score < 0.6) {
        return multi_point_crossover(parent1, parent2, 3);
    }
    else {
        size_t max_gates = std::max(parent1.gates.size(), parent2.gates.size());
        std::uniform_int_distribution<size_t> crossover_dist(0, max_gates);
        size_t crossover_point = crossover_dist(rng);
        
        for (size_t i = 0; i < crossover_point && i < parent1.gates.size(); ++i) {
            child.gates.push_back(parent1.gates[i]);
        }
        
        for (size_t i = crossover_point; i < parent2.gates.size(); ++i) {
            child.gates.push_back(parent2.gates[i]);
        }
    }
    
    return child;
}

QuantumCircuit HybridReversibleSboxBuilder::adaptive_mutate(
    const QuantumCircuit& circuit, 
    double base_mutation_rate,
    double diversity_score,
    int generation) {
    
    auto gates = build_gate_set();
    QuantumCircuit mutated_circuit = circuit.copy();
    
    double adjusted_mutation_rate = base_mutation_rate;
    
    if (diversity_score < 0.3) {
        adjusted_mutation_rate = std::min(0.8, base_mutation_rate * 2.0);
    }
    else if (diversity_score > 0.7) {
        adjusted_mutation_rate = std::max(0.05, base_mutation_rate * 0.8);
    }
    
    double generation_factor = std::max(0.5, 1.0 - (generation / 1000.0));
    adjusted_mutation_rate *= generation_factor;
    
    std::uniform_real_distribution<> random_real(0.0, 1.0);
    std::uniform_int_distribution<> gate_type_dist(0, gates.size() - 1);
    
    if (random_real(rng) < adjusted_mutation_rate) {
        auto gate = gates[gate_type_dist(rng)];
        
        std::vector<int> qubits(8);
        for (int i = 0; i < 8; ++i) {
            qubits[i] = i;
        }
        std::shuffle(qubits.begin(), qubits.end(), rng);
        
        if (qubits.size() >= static_cast<size_t>(gate.qubits)) {
            std::vector<int> selected_qubits(qubits.begin(), qubits.begin() + gate.qubits);
            gate.apply(mutated_circuit, selected_qubits);
        }
    }
    
    if (random_real(rng) < adjusted_mutation_rate * 0.7 && !mutated_circuit.gates.empty()) {
        std::uniform_int_distribution<size_t> gate_index_dist(0, mutated_circuit.gates.size() - 1);
        size_t gate_idx = gate_index_dist(rng);
        mutated_circuit.gates.erase(mutated_circuit.gates.begin() + gate_idx);
    }
    
    if (random_real(rng) < adjusted_mutation_rate * 0.5 && mutated_circuit.gates.size() > 1) {
        std::uniform_int_distribution<size_t> gate_index_dist(0, mutated_circuit.gates.size() - 1);
        size_t idx1 = gate_index_dist(rng);
        size_t idx2 = gate_index_dist(rng);
        
        if (idx1 != idx2) {
            std::swap(mutated_circuit.gates[idx1], mutated_circuit.gates[idx2]);
        }
    }
    
    if (random_real(rng) < 0.1) {
        mutated_circuit.optimize();
    }
    
    if (random_real(rng) < 0.05) {
        QuantumCircuit domain_circuit;
        
        if (random_real(rng) < 0.5) {
            domain_circuit = create_optimized_field_inversion_circuit();
        } else {
            domain_circuit = create_optimized_affine_transform_circuit();
        }
        
        int start = std::uniform_int_distribution<>(0, domain_circuit.gates.size() - 5)(rng);
        int length = std::uniform_int_distribution<>(3, 5)(rng);
        
        for (int i = start; i < start + length && i < (int)domain_circuit.gates.size(); i++) {
            mutated_circuit.gates.push_back(domain_circuit.gates[i]);
        }
    }
    
    return mutated_circuit;
}

QuantumCircuit HybridReversibleSboxBuilder::improved_adaptive_mutate(
    const QuantumCircuit& circuit, 
    double base_mutation_rate,
    double correctness_score,
    int generation) {
    
    auto gates = build_gate_set();
    QuantumCircuit mutated_circuit = circuit.copy();
    
    double adjusted_mutation_rate = base_mutation_rate;
    
    if (correctness_score > 0.7) {
        adjusted_mutation_rate *= 0.5;
    } else if (correctness_score < 0.3) {
        adjusted_mutation_rate *= 1.5;
    }
    
    double generation_factor = std::max(0.5, 1.0 - (generation / 2000.0));
    adjusted_mutation_rate *= generation_factor;
    
    std::uniform_real_distribution<> random_real(0.0, 1.0);
    
    bool has_field_inversion = false;
    bool has_affine_transform = false;
    
    for (size_t i = 0; i < mutated_circuit.gates.size(); i++) {
        if (i+3 < mutated_circuit.gates.size() &&
            mutated_circuit.gates[i].type == GateType::SWAP &&
            mutated_circuit.gates[i+1].type == GateType::CNOT &&
            mutated_circuit.gates[i+2].type == GateType::CNOT &&
            mutated_circuit.gates[i+3].type == GateType::TOFFOLI) {
            has_field_inversion = true;
        }
        
        if (i+7 < mutated_circuit.gates.size() &&
            mutated_circuit.gates[i].type == GateType::CNOT &&
            mutated_circuit.gates[i+1].type == GateType::CNOT &&
            mutated_circuit.gates[i+2].type == GateType::CNOT &&
            mutated_circuit.gates[i+3].type == GateType::CNOT &&
            mutated_circuit.gates[i+4].type == GateType::CNOT &&
            mutated_circuit.gates[i+5].type == GateType::CNOT &&
            mutated_circuit.gates[i+6].type == GateType::CNOT &&
            mutated_circuit.gates[i+7].type == GateType::NOT) {
            has_affine_transform = true;
        }
    }
    
    if (random_real(rng) < adjusted_mutation_rate) {
        if (!has_field_inversion && !has_affine_transform && random_real(rng) < 0.4) {
            if (random_real(rng) < 0.5) {
                QuantumCircuit inversion = create_optimized_field_inversion_circuit();
                mutated_circuit.append(inversion);
            } else {
                QuantumCircuit affine = create_optimized_affine_transform_circuit();
                mutated_circuit.append(affine);
            }
        } else {
            std::vector<double> gate_weights(gates.size(), 1.0);
            
            for (size_t i = 0; i < gates.size(); i++) {
                if (gates[i].name == "CNOT") {
                    gate_weights[i] = 2.0;
                } else if (gates[i].name == "Toffoli") {
                    gate_weights[i] = 1.5;
                }
            }
            
            std::discrete_distribution<> weighted_dist(gate_weights.begin(), gate_weights.end());
            auto gate = gates[weighted_dist(rng)];
            
            std::vector<int> qubits(8);
            for (int i = 0; i < 8; ++i) {
                qubits[i] = i;
            }
            std::shuffle(qubits.begin(), qubits.end(), rng);
            
            if (qubits.size() >= static_cast<size_t>(gate.qubits)) {
                std::vector<int> selected_qubits(qubits.begin(), qubits.begin() + gate.qubits);
                gate.apply(mutated_circuit, selected_qubits);
            }
        }
    }
    
    if (random_real(rng) < adjusted_mutation_rate * 0.5 && !mutated_circuit.gates.empty()) {
        std::uniform_int_distribution<size_t> gate_index_dist(0, mutated_circuit.gates.size() - 1);
        size_t gate_idx = gate_index_dist(rng);
        
        bool is_essential = false;
        
        if (gate_idx > 0 && gate_idx < mutated_circuit.gates.size() - 1) {
            if ((mutated_circuit.gates[gate_idx-1].type == GateType::SWAP && 
                 mutated_circuit.gates[gate_idx+1].type == GateType::CNOT) ||
                (mutated_circuit.gates[gate_idx-1].type == GateType::CNOT && 
                 mutated_circuit.gates[gate_idx+1].type == GateType::CNOT)) {
                is_essential = true;
            }
        }
        
        if (!is_essential) {
            mutated_circuit.gates.erase(mutated_circuit.gates.begin() + gate_idx);
        }
    }
    
    if (random_real(rng) < adjusted_mutation_rate * 0.3 && mutated_circuit.gates.size() > 1) {
        std::uniform_int_distribution<size_t> gate_index_dist(0, mutated_circuit.gates.size() - 1);
        size_t idx1 = gate_index_dist(rng);
        size_t idx2 = gate_index_dist(rng);
        
        if (idx1 != idx2) {
            std::swap(mutated_circuit.gates[idx1], mutated_circuit.gates[idx2]);
        }
    }
    
    if (random_real(rng) < 0.1 + (generation / 5000.0)) {
        mutated_circuit.optimize();
    }
    
    if (random_real(rng) < 0.05 + (correctness_score > 0.5 ? 0.0 : 0.1)) {
        QuantumCircuit full_solution = create_complete_aes_sbox_circuit();
        
        if (random_real(rng) < 0.3 || mutated_circuit.gates.empty()) {
            mutated_circuit = full_solution;
        } else {
            int start = std::uniform_int_distribution<>(0, full_solution.gates.size() - 5)(rng);
            int length = std::uniform_int_distribution<>(5, 10)(rng);
            
            for (int i = start; i < start + length && i < (int)full_solution.gates.size(); i++) {
                mutated_circuit.gates.push_back(full_solution.gates[i]);
            }
        }
    }
    
    return mutated_circuit;
}

int HybridReversibleSboxBuilder::tournament_selection(
    const std::vector<double>& scores, 
    double diversity_score) {
    
    std::uniform_int_distribution<> select_dist(0, scores.size() - 1);
    
    int tournament_size;
    if (diversity_score < 0.3) {
        tournament_size = 2;
    } else if (diversity_score < 0.6) {
        tournament_size = 3;
    } else {
        tournament_size = 4;
    }
    
    int best_index = select_dist(rng);
    double best_score = scores[best_index];
    
    for (int i = 1; i < tournament_size; i++) {
        int candidate_index = select_dist(rng);
        double candidate_score = scores[candidate_index];
        
        if (candidate_score > best_score) {
            best_index = candidate_index;
            best_score = candidate_score;
        }
    }
    
    return best_index;
}

QuantumCircuit HybridReversibleSboxBuilder::simulated_annealing_optimize(
    QuantumCircuit circuit, 
    int iterations, 
    double initial_temp) {
    
    auto gates = build_gate_set();
    QuantumCircuit best_circuit = circuit;
    double best_score = evaluate_circuit(circuit);
    
    double current_temp = initial_temp;
    double cooling_rate = 0.95;
    
    for (int i = 0; i < iterations; i++) {
        QuantumCircuit neighbor = circuit.copy();
        
        std::uniform_int_distribution<> mod_type(0, 2);
        int modification = mod_type(rng);
        
        if (modification == 0 && !neighbor.gates.empty()) {
            std::uniform_int_distribution<> gate_idx_dist(0, neighbor.gates.size() - 1);
            std::uniform_int_distribution<> gate_type_dist(0, gates.size() - 1);
            
            int idx = gate_idx_dist(rng);
            auto new_gate = gates[gate_type_dist(rng)];
            
            std::vector<int> qubits = {0, 1, 2, 3, 4, 5, 6, 7};
            std::shuffle(qubits.begin(), qubits.end(), rng);
            
            if (qubits.size() >= static_cast<size_t>(new_gate.qubits)) {
                neighbor.gates.erase(neighbor.gates.begin() + idx);
                new_gate.apply(neighbor, std::vector<int>(qubits.begin(), qubits.begin() + new_gate.qubits));
            }
        }
        else if (modification == 1 && neighbor.gates.size() > 1) {
            std::uniform_int_distribution<> gate_idx_dist(0, neighbor.gates.size() - 1);
            int idx1 = gate_idx_dist(rng);
            int idx2 = gate_idx_dist(rng);
            
            std::swap(neighbor.gates[idx1], neighbor.gates[idx2]);
        }
        else if (modification == 2) {
            if (!neighbor.gates.empty()) {
                std::uniform_int_distribution<> gate_idx_dist(0, neighbor.gates.size() - 1);
                int idx = gate_idx_dist(rng);
                
                if (neighbor.gates[idx].qubits.size() >= 2) {
                    std::swap(neighbor.gates[idx].qubits[0], neighbor.gates[idx].qubits[1]);
                }
            }
        }
        
        double neighbor_score = evaluate_circuit(neighbor);
        
        bool accept = false;
        
        if (neighbor_score > best_score) {
            best_circuit = neighbor;
            best_score = neighbor_score;
            accept = true;
        }
        
        if (!accept) {
            double delta = neighbor_score - evaluate_circuit(circuit);
            double acceptance_probability = exp(delta / current_temp);
            
            std::uniform_real_distribution<> prob_dist(0.0, 1.0);
            if (prob_dist(rng) < acceptance_probability) {
                accept = true;
            }
        }
        
        if (accept) {
            circuit = neighbor;
        }
        
        current_temp *= cooling_rate;
    }
    
    return best_circuit;
}

std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
HybridReversibleSboxBuilder::implement_aes_sbox_hybrid(
    int population_size, 
    int generations, 
    double mutation_rate) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto population = generate_initial_population(population_size, 200);
    
    std::vector<double> population_scores(population.size());
    for (size_t i = 0; i < population.size(); ++i) {
        population_scores[i] = evaluate_circuit(population[i]);
    }
    
    QuantumCircuit best_overall_circuit(8, 8);
    double best_overall_score = -std::numeric_limits<double>::infinity();
    
    int stagnation_counter = 0;
    double current_mutation_rate = mutation_rate;
    const int SA_ITERATIONS = 50;
    const double INITIAL_TEMP = 1.0;
    
    for (int generation = 0; generation < generations; ++generation) {
        double diversity_score = calculate_population_diversity(population);
        
        std::vector<int> sorted_indices(population.size());
        for (size_t i = 0; i < population.size(); ++i) {
            sorted_indices[i] = i;
        }
        
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return population_scores[a] > population_scores[b];
        });
        
        if (population_scores[sorted_indices[0]] > best_overall_score) {
            best_overall_score = population_scores[sorted_indices[0]];
            best_overall_circuit = population[sorted_indices[0]];
            
            if (verbose) {
                std::cout << "Generation " << generation 
                          << ": New best score " << best_overall_score 
                          << ", Diversity: " << diversity_score
                          << ", Mutation Rate: " << current_mutation_rate 
                          << std::endl;
            }
            
            stagnation_counter = 0;
        } else {
            stagnation_counter++;
        }
        
        if (generation % 50 == 0 && generation > 0) {
            QuantumCircuit optimized = simulated_annealing_optimize(
                best_overall_circuit, SA_ITERATIONS, INITIAL_TEMP);
            
            double optimized_score = evaluate_circuit(optimized);
            if (optimized_score > best_overall_score) {
                best_overall_score = optimized_score;
                best_overall_circuit = optimized;
                
                if (verbose) {
                    std::cout << "SA Optimization: New best score " << best_overall_score << std::endl;
                }
            }
        }
        
        if (stagnation_counter > 50) {
            if (verbose) {
                std::cout << "Restarting population due to stagnation" << std::endl;
            }
            
            int elite_count = population.size() * 0.1;
            std::vector<QuantumCircuit> elite_circuits;
            
            for (int i = 0; i < elite_count; i++) {
                elite_circuits.push_back(population[sorted_indices[i]]);
            }
            
            population = generate_initial_population(population_size - elite_count, 250);
            
            for (const auto& elite : elite_circuits) {
                population.push_back(elite);
            }
            
            population_scores.resize(population.size());
            for (size_t i = 0; i < population.size(); ++i) {
                population_scores[i] = evaluate_circuit(population[i]);
            }
            
            stagnation_counter = 0;
            current_mutation_rate = mutation_rate;
        }
        
        if (best_overall_score >= 0.99) {
            if (verbose) {
                std::cout << "Near-perfect S-box implementation achieved!" << std::endl;
            }
            break;
        }
        
        std::vector<QuantumCircuit> next_population;
        std::vector<double> next_scores;
        
        int num_elite = std::max(2, static_cast<int>(population.size() * 0.15));
        for (int i = 0; i < num_elite; ++i) {
            next_population.push_back(population[sorted_indices[i]]);
            next_scores.push_back(population_scores[sorted_indices[i]]);
        }
        
        while (next_population.size() < population.size()) {
            int parent1_idx = tournament_selection(population_scores, diversity_score);
            int parent2_idx = tournament_selection(population_scores, diversity_score);
            
            QuantumCircuit child = crossover_with_diversity(
                population[parent1_idx], 
                population[parent2_idx],
                diversity_score
            );
            
            child = adaptive_mutate(
                child, 
                current_mutation_rate, 
                diversity_score,
                generation
            );
            
            double child_score = evaluate_circuit(child);
            next_population.push_back(child);
            next_scores.push_back(child_score);
        }
        
        population = next_population;
        population_scores = next_scores;
        
        if (generation % 100 == 0 && generation > 0) {
            int replace_count = population.size() * 0.1;
            int start_idx = population.size() - replace_count;
            
            for (int i = 0; i < replace_count; i++) {
                if (i % 2 == 0) {
                    population[sorted_indices[start_idx + i]] = create_optimized_field_inversion_circuit();
                } else {
                    QuantumCircuit combined(8, 8);
                    combined.append(create_optimized_field_inversion_circuit());
                    combined.append(create_optimized_affine_transform_circuit());
                    population[sorted_indices[start_idx + i]] = combined;
                }
                
                population_scores[sorted_indices[start_idx + i]] = 
                    evaluate_circuit(population[sorted_indices[start_idx + i]]);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    if (verbose) {
        std::cout << "Optimization completed in " << duration << " seconds" << std::endl;
    }
    
    std::unordered_map<int, std::unordered_map<std::string, uint64_t>> test_results;
    
    for (int i = 0; i < 256; ++i) {
        uint64_t output = simulator.simulate(best_overall_circuit, i);
        uint64_t expected = AES_SBOX[i];
        
        test_results[i]["input"] = i;
        test_results[i]["output"] = output;
        test_results[i]["expected"] = expected;
        test_results[i]["correct"] = (output == expected) ? 1 : 0;
    }
    
    return {best_overall_circuit, test_results};
}

QuantumCircuit HybridReversibleSboxBuilder::direct_aes_sbox_construction() {
    QuantumCircuit circuit(8, 8);
    
    for (int i = 0; i < 4; i++) {
        circuit.swap(i, 7-i);
    }
    
    for (int i = 0; i < 4; i++) {
        for (int j = i+1; j < 4; j++) {
            circuit.cx(i, j);
        }
    }
    
    for (int i = 0; i < 3; i++) {
        circuit.ccx(i, i+1, i+2);
    }
    
    for (int i = 0; i < 3; i++) {
        circuit.swap(i, 6-i);
    }
    
    for (int i = 0; i < 8; i++) {
        circuit.cx((i+4) % 8, i);
        circuit.cx((i+5) % 8, i);
        circuit.cx((i+6) % 8, i);
        circuit.cx((i+7) % 8, i);
    }
    
    circuit.x(0);
    circuit.x(1);
    circuit.x(6);
    
    return circuit;
}

std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
HybridReversibleSboxBuilder::implement_aes_sbox(
    int population_size, int generations, double mutation_rate) {
    
    auto hybrid_result = implement_aes_sbox_hybrid(population_size, generations, mutation_rate);
    QuantumCircuit direct_circuit = direct_aes_sbox_construction();
    
    double hybrid_score = evaluate_circuit(hybrid_result.first);
    double direct_score = evaluate_circuit(direct_circuit);
    
    if (direct_score > hybrid_score) {
        if (verbose) {
            std::cout << "Direct construction performed better: " << direct_score << std::endl;
        }
        
        std::unordered_map<int, std::unordered_map<std::string, uint64_t>> test_results;
        for (int i = 0; i < 256; ++i) {
            uint64_t output = simulator.simulate(direct_circuit, i);
            uint64_t expected = AES_SBOX[i];
            
            test_results[i]["input"] = i;
            test_results[i]["output"] = output;
            test_results[i]["expected"] = expected;
            test_results[i]["correct"] = (output == expected) ? 1 : 0;
        }
        
        return {direct_circuit, test_results};
    } else {
        return hybrid_result;
    }
}

std::pair<QuantumCircuit, std::unordered_map<int, std::unordered_map<std::string, uint64_t>>> 
HybridReversibleSboxBuilder::implement_aes_sbox_improved(
    int population_size, 
    int generations, 
    double mutation_rate) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    QuantumCircuit direct_circuit = create_complete_aes_sbox_circuit();
    double direct_score = improved_evaluate_circuit(direct_circuit);
    
    if (direct_score > 0.95) {
        if (verbose) {
            std::cout << "Direct construction achieved excellent score: " << direct_score << std::endl;
        }
        
        direct_circuit.optimize();
        
        std::unordered_map<int, std::unordered_map<std::string, uint64_t>> test_results;
        for (int i = 0; i < 256; ++i) {
            uint64_t output = simulator.simulate(direct_circuit, i);
            uint64_t expected = AES_SBOX[i];
            
            test_results[i]["input"] = i;
            test_results[i]["output"] = output;
            test_results[i]["expected"] = expected;
            test_results[i]["correct"] = (output == expected) ? 1 : 0;
        }
        
        return {direct_circuit, test_results};
    }
    
    if (verbose) {
        std::cout << "Starting two-stage evolutionary optimization..." << std::endl;
    }
    
    QuantumCircuit best_inversion_circuit = create_optimized_field_inversion_circuit();
    if (verbose) {
        std::cout << "Starting field inversion optimization..." << std::endl;
    }
    
    auto inversion_population = generate_field_inversion_population(population_size);
    
    std::vector<double> inversion_scores(inversion_population.size());
    for (size_t i = 0; i < inversion_population.size(); ++i) {
        inversion_scores[i] = evaluate_field_inversion(inversion_population[i]);
    }
    
    for (int gen = 0; gen < generations / 2; ++gen) {
        double diversity = calculate_population_diversity(inversion_population);
        
        std::vector<QuantumCircuit> next_population;
        std::vector<double> next_scores;
        
        std::vector<size_t> sorted_indices(inversion_population.size());
        for (size_t i = 0; i < inversion_population.size(); ++i) {
            sorted_indices[i] = i;
        }
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                 [&](size_t a, size_t b) { return inversion_scores[a] > inversion_scores[b]; });
        
        int num_elite = std::max(2, static_cast<int>(inversion_population.size() * 0.1));
        for (int i = 0; i < num_elite; ++i) {
            next_population.push_back(inversion_population[sorted_indices[i]]);
            next_scores.push_back(inversion_scores[sorted_indices[i]]);
        }
        
        if (inversion_scores[sorted_indices[0]] > evaluate_field_inversion(best_inversion_circuit)) {
            best_inversion_circuit = inversion_population[sorted_indices[0]];
            if (verbose && gen % 10 == 0) {
                std::cout << "Generation " << gen 
                          << ": New best inversion score " << inversion_scores[sorted_indices[0]] 
                          << ", Diversity: " << diversity
                          << std::endl;
            }
        }
        
        while (next_population.size() < inversion_population.size()) {
            int parent1_idx = tournament_selection(inversion_scores, diversity);
            int parent2_idx = tournament_selection(inversion_scores, diversity);
            
            QuantumCircuit child = crossover_with_diversity(
                inversion_population[parent1_idx], 
                inversion_population[parent2_idx],
                diversity
            );
            
            child = improved_adaptive_mutate(
                child, 
                mutation_rate, 
                inversion_scores[sorted_indices[0]], 
                gen
            );
            
            double child_score = evaluate_field_inversion(child);
            next_population.push_back(child);
            next_scores.push_back(child_score);
        }
        
        inversion_population = next_population;
        inversion_scores = next_scores;
        
        if (gen % 20 == 0) {
            int replace_count = inversion_population.size() * 0.05;
            for (int i = 0; i < replace_count; i++) {
                inversion_population[inversion_population.size() - i - 1] = create_optimized_field_inversion_circuit();
                inversion_scores[inversion_population.size() - i - 1] = 
                    evaluate_field_inversion(inversion_population[inversion_population.size() - i - 1]);
            }
        }
    }
    
    if (verbose) {
        std::cout << "Starting complete S-box optimization with fixed inversion..." << std::endl;
    }
    
    auto population = generate_sbox_population_with_fixed_inversion(population_size, best_inversion_circuit);
    
    std::vector<double> population_scores(population.size());
    for (size_t i = 0; i < population.size(); ++i) {
        population_scores[i] = improved_evaluate_circuit(population[i]);
    }
    
    QuantumCircuit best_overall_circuit = direct_circuit;
    double best_overall_score = direct_score;
    
    for (int gen = 0; gen < generations / 2; ++gen) {
        double diversity = calculate_population_diversity(population);
        
        std::vector<size_t> sorted_indices(population.size());
        for (size_t i = 0; i < population.size(); ++i) {
            sorted_indices[i] = i;
        }
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                 [&](size_t a, size_t b) { return population_scores[a] > population_scores[b]; });
        
        if (population_scores[sorted_indices[0]] > best_overall_score) {
            best_overall_score = population_scores[sorted_indices[0]];
            best_overall_circuit = population[sorted_indices[0]];
            
            if (verbose && gen % 10 == 0) {
                std::cout << "Generation " << gen 
                          << ": New best overall score " << best_overall_score 
                          << ", Diversity: " << diversity
                          << std::endl;
            }
        }
        
        if (best_overall_score >= 0.99) {
            if (verbose) {
                std::cout << "Near-perfect S-box implementation achieved!" << std::endl;
            }
            break;
        }
        
        std::vector<QuantumCircuit> next_population;
        std::vector<double> next_scores;
        
        int num_elite = std::max(2, static_cast<int>(population.size() * 0.1));
        for (int i = 0; i < num_elite; ++i) {
            next_population.push_back(population[sorted_indices[i]]);
            next_scores.push_back(population_scores[sorted_indices[i]]);
        }
        
        while (next_population.size() < population.size()) {
            int parent1_idx = tournament_selection(population_scores, diversity);
            int parent2_idx = tournament_selection(population_scores, diversity);
            
            QuantumCircuit child = crossover_with_diversity(
                population[parent1_idx], 
                population[parent2_idx],
                diversity
            );
            
            child = improved_adaptive_mutate(
                child, 
                mutation_rate, 
                population_scores[sorted_indices[0]], 
                gen
            );
            
            double child_score = improved_evaluate_circuit(child);
            next_population.push_back(child);
            next_scores.push_back(child_score);
        }
        
        population = next_population;
        population_scores = next_scores;
        
        if (gen % 30 == 0) {
            int replace_count = population.size() * 0.05;
            for (int i = 0; i < replace_count; i++) {
                QuantumCircuit replacement = create_complete_aes_sbox_circuit();
                population[population.size() - i - 1] = replacement;
                population_scores[population.size() - i - 1] = improved_evaluate_circuit(replacement);
            }
        }
    }
    
    best_overall_circuit.optimize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    if (verbose) {
        std::cout << "Optimization completed in " << duration << " seconds" << std::endl;
        std::cout << "Final S-box score: " << best_overall_score << std::endl;
        std::cout << "Circuit size: " << best_overall_circuit.gates.size() << " gates" << std::endl;
    }
    
    std::unordered_map<int, std::unordered_map<std::string, uint64_t>> test_results;
    int correct_count = 0;
    
    for (int i = 0; i < 256; ++i) {
        uint64_t output = simulator.simulate(best_overall_circuit, i);
        uint64_t expected = AES_SBOX[i];
        
        test_results[i]["input"] = i;
        test_results[i]["output"] = output;
        test_results[i]["expected"] = expected;
        test_results[i]["correct"] = (output == expected) ? 1 : 0;
        
        if (output == expected) {
            correct_count++;
        }
    }
    
    if (verbose) {
        std::cout << "Correct outputs: " << correct_count << "/256 (" 
                  << (100.0 * correct_count / 256.0) << "%)" << std::endl;
    }
    
    return {best_overall_circuit, test_results};
}

double HybridReversibleSboxBuilder::calculate_entropy(const std::unordered_map<uint64_t, uint64_t>& mapping) {
    std::unordered_map<uint64_t, int> output_counts;
    
    for (const auto& pair : mapping) {
        output_counts[pair.second]++;
    }
    
    double entropy = 0.0;
    double total_inputs = mapping.size();
    
    for (const auto& count : output_counts) {
        double probability = count.second / total_inputs;
        entropy -= probability * std::log2(probability);
    }
    
    return entropy;
}

double HybridReversibleSboxBuilder::calculate_avalanche(const std::unordered_map<uint64_t, uint64_t>& mapping, int num_bits) {
    double total_avalanche = 0.0;
    int total_pairs = 0;
    
    for (uint64_t i = 0; i < (1ULL << num_bits); ++i) {
        for (int bit = 0; bit < num_bits; ++bit) {
            uint64_t j = i ^ (1ULL << bit);
            
            if (mapping.find(i) != mapping.end() && mapping.find(j) != mapping.end()) {
                uint64_t output_i = mapping.at(i);
                uint64_t output_j = mapping.at(j);
                uint64_t diff = output_i ^ output_j;
                
                int hamming_distance = 0;
                uint64_t temp = diff;
                while (temp) {
                    hamming_distance += temp & 1;
                    temp >>= 1;
                }
                
                total_avalanche += hamming_distance;
                total_pairs++;
            }
        }
    }
    
    return total_avalanche / (total_pairs * num_bits);
}

QuantumCircuit HybridReversibleSboxBuilder::crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2) {
    QuantumCircuit child = parent1.copy();
    child.gates.clear();
    
    size_t max_gates = std::max(parent1.gates.size(), parent2.gates.size());
    std::uniform_int_distribution<size_t> crossover_dist(0, max_gates);
    size_t crossover_point = crossover_dist(rng);
    
    for (size_t i = 0; i < crossover_point && i < parent1.gates.size(); ++i) {
        child.gates.push_back(parent1.gates[i]);
    }
    
    for (size_t i = crossover_point; i < parent2.gates.size(); ++i) {
        child.gates.push_back(parent2.gates[i]);
    }
    
    return child;
}

QuantumCircuit HybridReversibleSboxBuilder::multi_point_crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2, int num_points) {
    QuantumCircuit child = parent1.copy();
    child.gates.clear();
    
    size_t max_gates = std::max(parent1.gates.size(), parent2.gates.size());
    
    std::vector<size_t> crossover_points;
    for (int i = 0; i < num_points; ++i) {
        std::uniform_int_distribution<size_t> point_dist(0, max_gates);
        crossover_points.push_back(point_dist(rng));
    }
    
    std::sort(crossover_points.begin(), crossover_points.end());
    
    bool use_parent1 = true;
    size_t last_point = 0;
    for (size_t point : crossover_points) {
        const auto& parent = use_parent1 ? parent1 : parent2;
        
        for (size_t i = last_point; i < point && i < parent.gates.size(); ++i) {
            child.gates.push_back(parent.gates[i]);
        }
        
        use_parent1 = !use_parent1;
        last_point = point;
    }
    
    const auto& parent = use_parent1 ? parent1 : parent2;
    for (size_t i = last_point; i < parent.gates.size(); ++i) {
        child.gates.push_back(parent.gates[i]);
    }
    
    return child;
}

QuantumCircuit HybridReversibleSboxBuilder::uniform_crossover(const QuantumCircuit& parent1, const QuantumCircuit& parent2) {
    QuantumCircuit child = parent1.copy();
    child.gates.clear();
    
    size_t max_gates = std::max(parent1.gates.size(), parent2.gates.size());
    
    for (size_t i = 0; i < max_gates; ++i) {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        bool use_parent1 = dist(rng) < 0.5;
        
        if (use_parent1 && i < parent1.gates.size()) {
            child.gates.push_back(parent1.gates[i]);
        } else if (!use_parent1 && i < parent2.gates.size()) {
            child.gates.push_back(parent2.gates[i]);
        } else if (i < parent1.gates.size()) {
            child.gates.push_back(parent1.gates[i]);
        } else if (i < parent2.gates.size()) {
            child.gates.push_back(parent2.gates[i]);
        }
    }
    
    return child;
}