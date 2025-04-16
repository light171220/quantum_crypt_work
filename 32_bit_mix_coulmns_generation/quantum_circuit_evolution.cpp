#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <bitset>
#include <iomanip>
#include <unordered_set>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>

struct GateInfo {
    std::string gate_type;
    int q1;
    int q2;
    int q3;
    
    GateInfo(std::string type, int qubit1, int qubit2, int qubit3 = 0)
        : gate_type(type), q1(qubit1), q2(qubit2), q3(qubit3) {}
    
    GateInfo(const GateInfo& other)
        : gate_type(other.gate_type), q1(other.q1), q2(other.q2), q3(other.q3) {}
};

using CircuitConfig = std::vector<GateInfo>;
using TestVector = std::pair<uint32_t, uint32_t>;

class QuantumState {
private:
    uint32_t state;
    
public:
    QuantumState(uint32_t initial_state) : state(initial_state) {}
    
    void apply_cx(int control, int target) {
        if ((state & (1 << control)) != 0) {
            state ^= (1 << target);
        }
    }
    
    void apply_ccx(int control1, int control2, int target) {
        if ((state & (1 << control1)) != 0 && (state & (1 << control2)) != 0) {
            state ^= (1 << target);
        }
    }
    
    void apply_swap(int q1, int q2) {
        int bit1 = (state >> q1) & 1;
        int bit2 = (state >> q2) & 1;
        
        if (bit1 != bit2) {
            state ^= (1 << q1);
            state ^= (1 << q2);
        }
    }
    
    uint32_t measure() {
        return state;
    }
};

uint32_t classical_mix_columns(uint32_t state) {
    uint8_t b0 = (state >> 24) & 0xFF;
    uint8_t b1 = (state >> 16) & 0xFF;
    uint8_t b2 = (state >> 8) & 0xFF;
    uint8_t b3 = state & 0xFF;
    
    uint8_t new_b0 = b0 ^ b2;
    uint8_t new_b1 = b1 ^ b0 ^ b2;
    uint8_t new_b2 = b2 ^ b1;
    uint8_t new_b3 = b3;
    
    return (new_b0 << 24) | (new_b1 << 16) | (new_b2 << 8) | new_b3;
}

class MixColumnsQuantumCircuit {
private:
    uint32_t input_state;
    CircuitConfig circuit_config;
    
public:
    MixColumnsQuantumCircuit(uint32_t state, const CircuitConfig& config)
        : input_state(state), circuit_config(config) {}
    
    uint32_t execute_circuit() {
        QuantumState state(input_state);
        
        for (const auto& gate_info : circuit_config) {
            if (gate_info.gate_type == "cx") {
                state.apply_cx(gate_info.q1, gate_info.q2);
            } else if (gate_info.gate_type == "ccx") {
                state.apply_ccx(gate_info.q1, gate_info.q2, gate_info.q3);
            } else if (gate_info.gate_type == "swap") {
                state.apply_swap(gate_info.q1, gate_info.q2);
            }
        }
        
        return state.measure();
    }
};

CircuitConfig build_direct_mix_columns_circuit() {
    CircuitConfig circuit;
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", 8 + i, 24 + i));
    }
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", 24 + i, 16 + i));
    }
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", 8 + i, 16 + i));
    }
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", 16 + i, 8 + i));
    }
    
    return circuit;
}

std::vector<TestVector> generate_comprehensive_test_vectors(int num_random = 1000) {
    std::vector<TestVector> test_vectors;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
    
    for (int i = 0; i < num_random; i++) {
        uint32_t state = dist(gen);
        uint32_t result = classical_mix_columns(state);
        test_vectors.push_back({state, result});
    }
    
    std::vector<uint32_t> edge_cases = {
        0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA,
        0x0F0F0F0F, 0xF0F0F0F0, 0x00FF00FF, 0xFF00FF00,
        0x12345678, 0x87654321, 0xDEADBEEF, 0xC0FFEEEE
    };
    
    for (uint32_t case_val : edge_cases) {
        uint32_t result = classical_mix_columns(case_val);
        test_vectors.push_back({case_val, result});
    }
    
    return test_vectors;
}

class BruteForceGeneticAlgorithm {
private:
    int population_size;
    double mutation_rate;
    int max_gates;
    std::vector<CircuitConfig> population;
    std::random_device rd;
    std::mt19937 gen;
    std::vector<TestVector> training_vectors;
    std::vector<std::vector<TestVector>> batch_test_vectors;
    int num_threads;
    std::atomic<bool> perfect_solution_found;
    
    CircuitConfig generate_random_circuit_config() {
        std::uniform_int_distribution<> num_gates_dist(20, max_gates);
        std::uniform_int_distribution<> qubit_dist(0, 31);
        std::uniform_int_distribution<> gate_type_dist(0, 2);
        
        int num_gates = num_gates_dist(gen);
        CircuitConfig circuit_config;
        
        for (int i = 0; i < num_gates; i++) {
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            int q1 = 0, q2 = 0, q3 = 0;
            
            if (gate_idx == 0) {
                gate_type = "cx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
            } else if (gate_idx == 1) {
                gate_type = "ccx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                do {
                    q3 = qubit_dist(gen);
                } while (q3 == q1 || q3 == q2);
            } else {
                gate_type = "swap";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
            }
            
            circuit_config.push_back(GateInfo(gate_type, q1, q2, q3));
        }
        
        return circuit_config;
    }
    
    CircuitConfig generate_guided_circuit() {
        CircuitConfig base_circuit = build_direct_mix_columns_circuit();
        
        std::uniform_int_distribution<> mutation_num(0, 10);
        int num_mutations = mutation_num(gen);
        
        for (int i = 0; i < num_mutations; i++) {
            mutate_circuit(base_circuit);
        }
        
        return base_circuit;
    }
    
    void mutate_circuit(CircuitConfig& circuit) {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<> gate_type_dist(0, 2);
        std::uniform_int_distribution<> qubit_dist(0, 31);
        std::uniform_int_distribution<> idx_dist(0, circuit.size() - 1);
        
        if (prob_dist(gen) < 0.3 && circuit.size() < max_gates) {
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            int q1 = 0, q2 = 0, q3 = 0;
            
            if (gate_idx == 0) {
                gate_type = "cx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                circuit.push_back(GateInfo(gate_type, q1, q2, 0));
            } else if (gate_idx == 1) {
                gate_type = "ccx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                do {
                    q3 = qubit_dist(gen);
                } while (q3 == q1 || q3 == q2);
                circuit.push_back(GateInfo(gate_type, q1, q2, q3));
            } else {
                gate_type = "swap";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                circuit.push_back(GateInfo(gate_type, q1, q2, 0));
            }
        } else if (prob_dist(gen) < 0.3 && circuit.size() > 10) {
            int idx = idx_dist(gen);
            circuit.erase(circuit.begin() + idx);
        } else if (prob_dist(gen) < 0.7) {
            int idx = idx_dist(gen);
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            int q1 = 0, q2 = 0, q3 = 0;
            
            if (gate_idx == 0) {
                gate_type = "cx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                circuit[idx] = GateInfo(gate_type, q1, q2, 0);
            } else if (gate_idx == 1) {
                gate_type = "ccx";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                do {
                    q3 = qubit_dist(gen);
                } while (q3 == q1 || q3 == q2);
                circuit[idx] = GateInfo(gate_type, q1, q2, q3);
            } else {
                gate_type = "swap";
                q1 = qubit_dist(gen);
                do {
                    q2 = qubit_dist(gen);
                } while (q1 == q2);
                circuit[idx] = GateInfo(gate_type, q1, q2, 0);
            }
        }
    }
    
    void prepare_batches() {
        int batch_size = training_vectors.size() / num_threads;
        if (batch_size == 0) batch_size = 1;
        
        batch_test_vectors.clear();
        
        for (int i = 0; i < num_threads; i++) {
            int start_idx = i * batch_size;
            int end_idx = (i == num_threads - 1) ? training_vectors.size() : (i + 1) * batch_size;
            
            std::vector<TestVector> batch(training_vectors.begin() + start_idx, 
                                        training_vectors.begin() + end_idx);
            batch_test_vectors.push_back(batch);
        }
    }
    
    std::vector<double> parallel_fitness_evaluation() {
        std::vector<double> fitness_scores(population.size(), 0.0);
        std::vector<std::future<void>> futures;
        
        int circuits_per_thread = population.size() / num_threads;
        if (circuits_per_thread == 0) circuits_per_thread = 1;
        
        for (int t = 0; t < num_threads; t++) {
            int start_idx = t * circuits_per_thread;
            int end_idx = (t == num_threads - 1) ? population.size() : (t + 1) * circuits_per_thread;
            
            futures.push_back(std::async(std::launch::async, [this, start_idx, end_idx, &fitness_scores]() {
                for (int i = start_idx; i < end_idx; i++) {
                    int passed_tests = 0;
                    int total_tests = 0;
                    
                    for (const auto& batch : batch_test_vectors) {
                        for (const auto& test_vec : batch) {
                            uint32_t input_state = test_vec.first;
                            uint32_t classical_output = test_vec.second;
                            
                            MixColumnsQuantumCircuit qc_instance(input_state, population[i]);
                            uint32_t quantum_output = qc_instance.execute_circuit();
                            
                            if (quantum_output == classical_output) {
                                passed_tests++;
                            }
                            total_tests++;
                        }
                    }
                    
                    fitness_scores[i] = static_cast<double>(passed_tests) / total_tests;
                    
                    if (fitness_scores[i] >= 0.999) {
                        perfect_solution_found = true;
                    }
                }
            }));
        }
        
        for (auto& future : futures) {
            future.get();
        }
        
        return fitness_scores;
    }
    
public:
    BruteForceGeneticAlgorithm(int pop_size = 100, double mut_rate = 0.3, int max_g = 50, int threads = 8)
        : population_size(pop_size), mutation_rate(mut_rate), max_gates(max_g), 
          gen(rd()), num_threads(threads), perfect_solution_found(false) {
        
        training_vectors = generate_comprehensive_test_vectors(500);
        prepare_batches();
        
        population.push_back(build_direct_mix_columns_circuit());
        
        for (int i = 1; i < population_size / 2; i++) {
            population.push_back(generate_guided_circuit());
        }
        
        for (int i = population_size / 2; i < population_size; i++) {
            population.push_back(generate_random_circuit_config());
        }
    }
    
    std::vector<CircuitConfig> selection(const std::vector<double>& fitness_scores) {
        std::vector<CircuitConfig> selected;
        
        auto max_it = std::max_element(fitness_scores.begin(), fitness_scores.end());
        int best_idx = std::distance(fitness_scores.begin(), max_it);
        selected.push_back(population[best_idx]);
        
        std::uniform_int_distribution<> idx_dist(0, population_size - 1);
        
        while (selected.size() < population_size) {
            std::vector<int> tournament;
            for (int j = 0; j < 5; j++) {
                tournament.push_back(idx_dist(gen));
            }
            
            int winner = tournament[0];
            for (int j = 1; j < 5; j++) {
                if (fitness_scores[tournament[j]] > fitness_scores[winner]) {
                    winner = tournament[j];
                }
            }
            
            selected.push_back(population[winner]);
        }
        
        return selected;
    }
    
    std::vector<CircuitConfig> crossover(const std::vector<CircuitConfig>& selected_population) {
        std::vector<CircuitConfig> new_population;
        new_population.push_back(selected_population[0]);
        
        std::uniform_int_distribution<> parent_dist(0, selected_population.size() - 1);
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        
        while (new_population.size() < population_size) {
            int parent1_idx = parent_dist(gen);
            int parent2_idx = parent_dist(gen);
            
            while (parent2_idx == parent1_idx) {
                parent2_idx = parent_dist(gen);
            }
            
            const CircuitConfig& parent1 = selected_population[parent1_idx];
            const CircuitConfig& parent2 = selected_population[parent2_idx];
            
            if (prob_dist(gen) < 0.8) {
                CircuitConfig child;
                std::uniform_int_distribution<> crossover_point_dist(1, std::min(parent1.size(), parent2.size()) - 1);
                
                if (std::min(parent1.size(), parent2.size()) > 1) {
                    int crossover_point = crossover_point_dist(gen);
                    
                    for (int i = 0; i < crossover_point && i < parent1.size(); i++) {
                        child.push_back(parent1[i]);
                    }
                    
                    for (int i = crossover_point; i < parent2.size(); i++) {
                        child.push_back(parent2[i]);
                    }
                } else {
                    child = parent1;
                }
                
                if (child.empty()) {
                    child = generate_random_circuit_config();
                }
                
                new_population.push_back(child);
            } else {
                new_population.push_back(prob_dist(gen) < 0.5 ? parent1 : parent2);
            }
        }
        
        return new_population;
    }
    
    std::vector<CircuitConfig> mutation(std::vector<CircuitConfig> population) {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        
        for (int i = 1; i < population.size(); i++) {
            if (prob_dist(gen) < mutation_rate) {
                mutate_circuit(population[i]);
            }
        }
        
        return population;
    }
    
    std::pair<CircuitConfig, double> evolve(int generations = 100) {
        double best_fitness = 0.0;
        CircuitConfig best_circuit_config;
        
        for (int generation = 0; generation < generations; generation++) {
            if (perfect_solution_found) {
                std::cout << "Perfect solution found at generation " << generation << "!" << std::endl;
                break;
            }
            
            std::vector<double> fitness_scores = parallel_fitness_evaluation();
            
            auto max_it = std::max_element(fitness_scores.begin(), fitness_scores.end());
            double current_best_fitness = *max_it;
            int best_idx = std::distance(fitness_scores.begin(), max_it);
            
            if (current_best_fitness > best_fitness) {
                best_fitness = current_best_fitness;
                best_circuit_config = population[best_idx];
                
                std::cout << "Generation " << (generation + 1) << ": Best Fitness = " 
                          << std::fixed << std::setprecision(4) << best_fitness;
                
                if (best_fitness >= 0.999) {
                    std::cout << " - Perfect solution found!" << std::endl;
                    break;
                }
                std::cout << std::endl;
            }
            
            if (generation % 10 == 0) {
                std::cout << "Generation " << (generation + 1) << ": Best Fitness = " 
                          << std::fixed << std::setprecision(4) << best_fitness << std::endl;
            }
            
            std::vector<CircuitConfig> selected_population = selection(fitness_scores);
            population = crossover(selected_population);
            population = mutation(population);
        }
        
        return {best_circuit_config, best_fitness};
    }
};

void test_circuit(const CircuitConfig& circuit, const std::vector<TestVector>& test_vectors) {
    std::cout << "Testing Circuit:" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    int total_tests = test_vectors.size();
    int passed_tests = 0;
    std::vector<std::map<std::string, uint32_t>> failed_tests;
    
    for (const auto& test_vec : test_vectors) {
        uint32_t input_state = test_vec.first;
        uint32_t classical_output = test_vec.second;
        
        MixColumnsQuantumCircuit qc_instance(input_state, circuit);
        uint32_t quantum_output = qc_instance.execute_circuit();
        
        if (quantum_output == classical_output) {
            passed_tests++;
        } else {
            std::map<std::string, uint32_t> failed_test;
            failed_test["input"] = input_state;
            failed_test["classical_output"] = classical_output;
            failed_test["quantum_output"] = quantum_output;
            failed_tests.push_back(failed_test);
        }
    }
    
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "Total Tests:  " << total_tests << std::endl;
    std::cout << "Passed Tests: " << passed_tests << std::endl;
    std::cout << "Failed Tests: " << (total_tests - passed_tests) << std::endl;
    std::cout << "Success Rate: " << std::fixed << std::setprecision(2) 
              << (static_cast<double>(passed_tests) / total_tests * 100.0) << "%" << std::endl;
    
    if (!failed_tests.empty()) {
        std::ofstream csvfile("failed_mix_columns_tests.csv");
        csvfile << "Input (Hex),Classical Output (Hex),Quantum Output (Hex)" << std::endl;
        
        int count = 0;
        for (const auto& test : failed_tests) {
            if (count >= 100) break;
            
            csvfile << "0x" << std::hex << std::setw(8) << std::setfill('0') << test.at("input") << ","
                    << "0x" << std::hex << std::setw(8) << std::setfill('0') << test.at("classical_output") << ","
                    << "0x" << std::hex << std::setw(8) << std::setfill('0') << test.at("quantum_output") << std::endl;
            count++;
        }
        
        csvfile.close();
        std::cout << "\nFirst 100 failed tests saved to 'failed_mix_columns_tests.csv'" << std::endl;
    }
    
    std::ofstream jsonfile("best_circuit_config.json");
    jsonfile << "[\n";
    for (size_t i = 0; i < circuit.size(); i++) {
        const auto& gate = circuit[i];
        jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
        if (gate.gate_type == "ccx") {
            jsonfile << ", " << gate.q3;
        } else {
            jsonfile << ", 0";
        }
        jsonfile << "]";
        if (i < circuit.size() - 1) jsonfile << ",";
        jsonfile << "\n";
    }
    jsonfile << "]\n";
    jsonfile.close();
}

int main() {
    std::cout << "Starting optimized parameter exploration..." << std::endl;
    
    struct CircuitWithFitness {
        CircuitConfig circuit;
        double fitness;
        
        // Add a default constructor
        CircuitWithFitness() : circuit(), fitness(0.0) {}
        
        CircuitWithFitness(CircuitConfig c, double f) : circuit(c), fitness(f) {}
        
        bool operator<(const CircuitWithFitness& other) const {
            return fitness > other.fitness; // For descending order
        }
    };
    
    std::vector<CircuitWithFitness> top_circuits;
    double best_overall_fitness = 0.0;
    
    // Define promising parameter combinations to try
    // Format: {population, mutation_rate, max_gates, generations}
    std::vector<std::tuple<int, double, int, int>> parameter_sets = {
        // Small, fast explorations
        {100, 0.1, 40, 8000},    // Small population, low mutation, fewer gates
        {150, 0.2, 60, 10000},   // Medium balanced approach
        
        // Medium-sized explorations
        {200, 0.15, 70, 12000},  // Medium population with controlled mutation
        {250, 0.25, 80, 12000},  // Larger population with higher mutation
        
        // Large, thorough explorations
        {300, 0.2, 90, 15000},   // Large population, moderate mutation
        {300, 0.3, 100, 15000},  // Large population, high mutation, max gates
        
        // Specialized trials
        {200, 0.1, 100, 12000},  // Focus on complex circuits with careful mutation
        {300, 0.15, 60, 12000}   // Large population but simpler circuits
    };
    
    for (const auto& params : parameter_sets) {
        int population = std::get<0>(params);
        double mutation_rate = std::get<1>(params);
        int max_gates = std::get<2>(params);
        int generations = std::get<3>(params);
        
        std::cout << "\n=== Testing parameters: population=" << population 
                  << ", mutation_rate=" << mutation_rate 
                  << ", max_gates=" << max_gates 
                  << ", generations=" << generations << " ===" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        BruteForceGeneticAlgorithm genetic_algo(population, mutation_rate, max_gates, 16);
        auto [best_circuit, best_fitness] = genetic_algo.evolve(generations);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "Run completed in " << duration << " seconds" << std::endl;
        std::cout << "Best fitness: " << best_fitness << std::endl;
        
        // Add to top circuits if good enough
        top_circuits.push_back(CircuitWithFitness(best_circuit, best_fitness));
        
        // Sort and keep only top 20
        std::sort(top_circuits.begin(), top_circuits.end());
        if (top_circuits.size() > 20) {
            top_circuits.resize(20);
        }
        
        if (best_fitness > best_overall_fitness) {
            best_overall_fitness = best_fitness;
            
            std::cout << "New best fitness found: " << best_fitness << std::endl;
            
            if (best_fitness >= 0.999) {
                std::cout << "Perfect solution found! Stopping search." << std::endl;
                goto perfect_solution_found;
            }
        }
    }
    
perfect_solution_found:
    std::cout << "\n=== Top 20 best solutions ===" << std::endl;
    
    for (size_t i = 0; i < top_circuits.size(); i++) {
        std::cout << "Circuit #" << (i+1) << " - Fitness: " << top_circuits[i].fitness << std::endl;
        
        // Save each circuit to a separate JSON file
        std::string filename = "circuit_" + std::to_string(i+1) + "_fitness_" + 
                              std::to_string(top_circuits[i].fitness).substr(0, 6) + ".json";
        
        std::ofstream jsonfile(filename);
        jsonfile << "[\n";
        for (size_t j = 0; j < top_circuits[i].circuit.size(); j++) {
            const auto& gate = top_circuits[i].circuit[j];
            jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
            if (gate.gate_type == "ccx") {
                jsonfile << ", " << gate.q3;
            } else {
                jsonfile << ", 0";
            }
            jsonfile << "]";
            if (j < top_circuits[i].circuit.size() - 1) jsonfile << ",";
            jsonfile << "\n";
        }
        jsonfile << "]\n";
        jsonfile.close();
    }
    
    // Test the very best circuit
    std::cout << "\n=== Testing the best solution (fitness: " << 
        (top_circuits.empty() ? 0.0 : top_circuits[0].fitness) << ") ===" << std::endl;
    
    std::vector<TestVector> final_test_vectors = generate_comprehensive_test_vectors(5000);
    
    if (!top_circuits.empty()) {
        test_circuit(top_circuits[0].circuit, final_test_vectors);
    } else {
        std::cout << "No circuits found!" << std::endl;
    }
    
    return 0;
}