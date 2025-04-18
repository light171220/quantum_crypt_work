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

const std::vector<uint8_t> AES_SBOX = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

struct GateInfo {
    std::string gate_type;
    int q1;
    int q2;
    int q3;
    
    GateInfo(std::string type, int qubit1, int qubit2 = -1, int qubit3 = -1)
        : gate_type(type), q1(qubit1), q2(qubit2), q3(qubit3) {}
    
    GateInfo(const GateInfo& other)
        : gate_type(other.gate_type), q1(other.q1), q2(other.q2), q3(other.q3) {}
};

using CircuitConfig = std::vector<GateInfo>;
using TestVector = std::pair<uint8_t, uint8_t>;

class QuantumState {
private:
    uint32_t state;
    
public:
    QuantumState(uint32_t initial_state) : state(initial_state) {}
    
    void apply_x(int target) {
        state ^= (1 << target);
    }
    
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

uint8_t classical_sbox_lookup(uint8_t input) {
    return AES_SBOX[input];
}

class SBoxQuantumCircuit {
private:
    uint8_t input_state;
    CircuitConfig circuit_config;
    int input_qubits;
    int output_qubits;
    
public:
    SBoxQuantumCircuit(uint8_t state, const CircuitConfig& config, int in_qubits = 8, int out_qubits = 8)
        : input_state(state), circuit_config(config), input_qubits(in_qubits), output_qubits(out_qubits) {}
    
    uint8_t execute_circuit() {
        uint32_t full_state = input_state;
        QuantumState qstate(full_state);
        
        for (const auto& gate_info : circuit_config) {
            if (gate_info.gate_type == "x") {
                qstate.apply_x(gate_info.q1);
            } else if (gate_info.gate_type == "cx") {
                qstate.apply_cx(gate_info.q1, gate_info.q2);
            } else if (gate_info.gate_type == "ccx") {
                qstate.apply_ccx(gate_info.q1, gate_info.q2, gate_info.q3);
            } else if (gate_info.gate_type == "swap") {
                qstate.apply_swap(gate_info.q1, gate_info.q2);
            }
        }
        
        uint32_t final_state = qstate.measure();
        return (final_state >> input_qubits) & 0xFF;
    }
};

CircuitConfig build_direct_sbox_circuit() {
    CircuitConfig circuit;
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", i, i + 8));
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = i+1; j < 8; j++) {
            circuit.push_back(GateInfo("cx", 8 + i, 8 + j));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        if ((0x63 >> i) & 1) {
            circuit.push_back(GateInfo("x", 8 + i));
        }
    }
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", 8 + ((i + 1) % 8), 8 + i));
        circuit.push_back(GateInfo("cx", 8 + ((i + 3) % 8), 8 + i));
        circuit.push_back(GateInfo("cx", 8 + ((i + 4) % 8), 8 + i));
    }
    
    return circuit;
}

CircuitConfig build_swap_based_sbox_circuit() {
    CircuitConfig circuit;
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", i, i + 8));
    }
    
    for (int i = 0; i < 4; i++) {
        circuit.push_back(GateInfo("swap", 8 + i, 8 + (7 - i)));
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                circuit.push_back(GateInfo("cx", 8 + i, 8 + j));
            }
        }
    }
    
    for (int i = 0; i < 8; i++) {
        if ((0x63 >> i) & 1) {
            circuit.push_back(GateInfo("x", 8 + i));
        }
    }
    
    return circuit;
}

CircuitConfig build_compact_sbox_circuit() {
    CircuitConfig circuit;
    
    for (int i = 0; i < 8; i++) {
        circuit.push_back(GateInfo("cx", i, i + 8));
    }
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                int parity = (i ^ j) & 0x7;
                if (parity == 1 || parity == 2 || parity == 4) {
                    circuit.push_back(GateInfo("cx", 8 + i, 8 + j));
                }
            }
        }
    }
    
    for (int i = 0; i < 8; i++) {
        if ((0x63 >> i) & 1) {
            circuit.push_back(GateInfo("x", 8 + i));
        }
    }
    
    return circuit;
}

std::vector<TestVector> generate_comprehensive_test_vectors() {
    std::vector<TestVector> test_vectors;
    
    for (int i = 0; i < 256; i++) {
        uint8_t input = static_cast<uint8_t>(i);
        uint8_t output = classical_sbox_lookup(input);
        test_vectors.push_back({input, output});
    }
    
    return test_vectors;
}

CircuitConfig simplify_circuit(const CircuitConfig& circuit) {
    CircuitConfig simplified = circuit;
    
    for (size_t i = 0; i < simplified.size() - 1; i++) {
        if (simplified[i].gate_type == "x" && 
            i + 1 < simplified.size() && 
            simplified[i+1].gate_type == "x" && 
            simplified[i].q1 == simplified[i+1].q1) {
            
            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }
    
    for (size_t i = 0; i < simplified.size() - 1; i++) {
        if (simplified[i].gate_type == "cx" && 
            i + 1 < simplified.size() && 
            simplified[i+1].gate_type == "cx" && 
            simplified[i].q1 == simplified[i+1].q1 &&
            simplified[i].q2 == simplified[i+1].q2) {
            
            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }
    
    for (size_t i = 0; i < simplified.size() - 1; i++) {
        if (simplified[i].gate_type == "ccx" && 
            i + 1 < simplified.size() && 
            simplified[i+1].gate_type == "ccx" && 
            simplified[i].q1 == simplified[i+1].q1 &&
            simplified[i].q2 == simplified[i+1].q2 &&
            simplified[i].q3 == simplified[i+1].q3) {
            
            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }
    
    for (size_t i = 0; i < simplified.size() - 1; i++) {
        if (simplified[i].gate_type == "swap" && 
            i + 1 < simplified.size() && 
            simplified[i+1].gate_type == "swap" && 
            ((simplified[i].q1 == simplified[i+1].q1 &&
              simplified[i].q2 == simplified[i+1].q2) ||
             (simplified[i].q1 == simplified[i+1].q2 &&
              simplified[i].q2 == simplified[i+1].q1))) {
            
            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }
    
    std::map<int, bool> qubit_affects_output;
    for (int i = 8; i < 16; i++) {
        qubit_affects_output[i] = true;
    }
    
    for (int i = simplified.size() - 1; i >= 0; i--) {
        const auto& gate = simplified[i];
        
        if (gate.gate_type == "x") {
            if (!qubit_affects_output[gate.q1]) {
                simplified.erase(simplified.begin() + i);
            }
        } else if (gate.gate_type == "cx") {
            if (qubit_affects_output[gate.q2]) {
                qubit_affects_output[gate.q1] = true;
            } else {
                simplified.erase(simplified.begin() + i);
            }
        } else if (gate.gate_type == "ccx") {
            if (qubit_affects_output[gate.q3]) {
                qubit_affects_output[gate.q1] = true;
                qubit_affects_output[gate.q2] = true;
            } else {
                simplified.erase(simplified.begin() + i);
            }
        } else if (gate.gate_type == "swap") {
            if (qubit_affects_output[gate.q1] || qubit_affects_output[gate.q2]) {
                qubit_affects_output[gate.q1] = true;
                qubit_affects_output[gate.q2] = true;
            } else {
                simplified.erase(simplified.begin() + i);
            }
        }
    }
    
    return simplified;
}

void test_circuit(const CircuitConfig& circuit, const std::vector<TestVector>& test_vectors) {
    std::cout << "Testing Circuit:" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    int total_tests = test_vectors.size();
    int passed_tests = 0;
    std::vector<std::map<std::string, uint8_t>> failed_tests;
    
    for (const auto& test_vec : test_vectors) {
        uint8_t input_state = test_vec.first;
        uint8_t classical_output = test_vec.second;
        
        SBoxQuantumCircuit qc_instance(input_state, circuit);
        uint8_t quantum_output = qc_instance.execute_circuit();
        
        if (quantum_output == classical_output) {
            passed_tests++;
        } else {
            std::map<std::string, uint8_t> failed_test;
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
    
    if (!failed_tests.empty() && failed_tests.size() <= 100) {
        std::ofstream csvfile("failed_sbox_tests.csv");
        csvfile << "Input (Hex),Classical Output (Hex),Quantum Output (Hex)" << std::endl;
        
        for (const auto& test : failed_tests) {
            csvfile << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("input")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("classical_output")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("quantum_output")) << std::endl;
        }
        
        csvfile.close();
        std::cout << "\nFailed tests saved to 'failed_sbox_tests.csv'" << std::endl;
    } else if (failed_tests.size() > 100) {
        std::ofstream csvfile("failed_sbox_tests.csv");
        csvfile << "Input (Hex),Classical Output (Hex),Quantum Output (Hex)" << std::endl;
        
        for (int i = 0; i < 100; i++) {
            const auto& test = failed_tests[i];
            csvfile << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("input")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("classical_output")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("quantum_output")) << std::endl;
        }
        
        csvfile.close();
        std::cout << "\nFirst 100 failed tests saved to 'failed_sbox_tests.csv'" << std::endl;
    }
    
    std::ofstream jsonfile("best_sbox_circuit_config.json");
    jsonfile << "[\n";
    for (size_t i = 0; i < circuit.size(); i++) {
        const auto& gate = circuit[i];
        jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
        if (gate.gate_type == "ccx") {
            jsonfile << ", " << gate.q3;
        } else {
            jsonfile << ", -1";
        }
        jsonfile << "]";
        if (i < circuit.size() - 1) jsonfile << ",";
        jsonfile << "\n";
    }
    jsonfile << "]\n";
    jsonfile.close();
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
        std::uniform_int_distribution<> num_gates_dist(50, max_gates);
        std::uniform_int_distribution<> qubit_dist(0, 15);
        std::uniform_int_distribution<> gate_type_dist(0, 3);
        std::uniform_int_distribution<> output_qubit_dist(8, 15);
        
        int num_gates = num_gates_dist(gen);
        CircuitConfig circuit_config;
        
        for (int i = 0; i < num_gates; i++) {
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            
            if (gate_idx == 0) {
                gate_type = "x";
                int target = qubit_dist(gen);
                circuit_config.push_back(GateInfo(gate_type, target));
            } else if (gate_idx == 1) {
                gate_type = "cx";
                int control = qubit_dist(gen);
                int target = qubit_dist(gen);
                while (target == control) {
                    target = qubit_dist(gen);
                }
                circuit_config.push_back(GateInfo(gate_type, control, target));
            } else if (gate_idx == 2) {
                gate_type = "ccx";
                int control1 = qubit_dist(gen);
                int control2 = qubit_dist(gen);
                while (control2 == control1) {
                    control2 = qubit_dist(gen);
                }
                int target = qubit_dist(gen);
                while (target == control1 || target == control2) {
                    target = qubit_dist(gen);
                }
                circuit_config.push_back(GateInfo(gate_type, control1, control2, target));
            } else {
                gate_type = "swap";
                int q1 = qubit_dist(gen);
                int q2 = qubit_dist(gen);
                while (q2 == q1) {
                    q2 = qubit_dist(gen);
                }
                circuit_config.push_back(GateInfo(gate_type, q1, q2));
            }
        }
        
        return circuit_config;
    }
    
    void mutate_circuit(CircuitConfig& circuit) {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<> gate_type_dist(0, 3);
        std::uniform_int_distribution<> qubit_dist(0, 15);
        std::uniform_int_distribution<> idx_dist(0, circuit.size() - 1);
        
        if (prob_dist(gen) < 0.3 && circuit.size() < max_gates) {
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            
            if (gate_idx == 0) {
                gate_type = "x";
                int target = qubit_dist(gen);
                circuit.push_back(GateInfo(gate_type, target));
            } else if (gate_idx == 1) {
                gate_type = "cx";
                int control = qubit_dist(gen);
                int target = qubit_dist(gen);
                while (target == control) {
                    target = qubit_dist(gen);
                }
                circuit.push_back(GateInfo(gate_type, control, target));
            } else if (gate_idx == 2) {
                gate_type = "ccx";
                int control1 = qubit_dist(gen);
                int control2 = qubit_dist(gen);
                while (control2 == control1) {
                    control2 = qubit_dist(gen);
                }
                int target = qubit_dist(gen);
                while (target == control1 || target == control2) {
                    target = qubit_dist(gen);
                }
                circuit.push_back(GateInfo(gate_type, control1, control2, target));
            } else {
                gate_type = "swap";
                int q1 = qubit_dist(gen);
                int q2 = qubit_dist(gen);
                while (q2 == q1) {
                    q2 = qubit_dist(gen);
                }
                circuit.push_back(GateInfo(gate_type, q1, q2));
            }
        } else if (prob_dist(gen) < 0.3 && circuit.size() > 10) {
            int idx = idx_dist(gen);
            circuit.erase(circuit.begin() + idx);
        } else if (prob_dist(gen) < 0.7) {
            int idx = idx_dist(gen);
            int gate_idx = gate_type_dist(gen);
            std::string gate_type;
            
            if (gate_idx == 0) {
                gate_type = "x";
                int target = qubit_dist(gen);
                circuit[idx] = GateInfo(gate_type, target);
            } else if (gate_idx == 1) {
                gate_type = "cx";
                int control = qubit_dist(gen);
                int target = qubit_dist(gen);
                while (target == control) {
                    target = qubit_dist(gen);
                }
                circuit[idx] = GateInfo(gate_type, control, target);
            } else if (gate_idx == 2) {
                gate_type = "ccx";
                int control1 = qubit_dist(gen);
                int control2 = qubit_dist(gen);
                while (control2 == control1) {
                    control2 = qubit_dist(gen);
                }
                int target = qubit_dist(gen);
                while (target == control1 || target == control2) {
                    target = qubit_dist(gen);
                }
                circuit[idx] = GateInfo(gate_type, control1, control2, target);
            } else {
                gate_type = "swap";
                int q1 = qubit_dist(gen);
                int q2 = qubit_dist(gen);
                while (q2 == q1) {
                    q2 = qubit_dist(gen);
                }
                circuit[idx] = GateInfo(gate_type, q1, q2);
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
                            uint8_t input_state = test_vec.first;
                            uint8_t classical_output = test_vec.second;
                            
                            SBoxQuantumCircuit qc_instance(input_state, population[i]);
                            uint8_t quantum_output = qc_instance.execute_circuit();
                            
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
    BruteForceGeneticAlgorithm(int pop_size = 100, double mut_rate = 0.3, int max_g = 200, int threads = 8)
        : population_size(pop_size), mutation_rate(mut_rate), max_gates(max_g), 
          gen(rd()), num_threads(threads), perfect_solution_found(false) {
        
        training_vectors = generate_comprehensive_test_vectors();
        prepare_batches();
        
        population.push_back(build_direct_sbox_circuit());
        population.push_back(build_swap_based_sbox_circuit());
        population.push_back(build_compact_sbox_circuit());
        
        for (int i = 3; i < population_size / 3; i++) {
            CircuitConfig base = build_direct_sbox_circuit();
            std::uniform_int_distribution<> mutation_num(1, 10);
            int num_mutations = mutation_num(gen);
            
            for (int j = 0; j < num_mutations; j++) {
                mutate_circuit(base);
            }
            population.push_back(base);
        }
        
        for (int i = population_size / 3; i < population_size * 2 / 3; i++) {
            CircuitConfig base = build_swap_based_sbox_circuit();
            std::uniform_int_distribution<> mutation_num(1, 10);
            int num_mutations = mutation_num(gen);
            
            for (int j = 0; j < num_mutations; j++) {
                mutate_circuit(base);
            }
            population.push_back(base);
        }
        
        for (int i = population_size * 2 / 3; i < population_size; i++) {
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

int main() {
    std::cout << "Starting AES S-box Quantum Circuit Generation..." << std::endl;
    
    struct CircuitWithFitness {
        CircuitConfig circuit;
        double fitness;
        
        CircuitWithFitness() : circuit(), fitness(0.0) {}
        CircuitWithFitness(CircuitConfig c, double f) : circuit(c), fitness(f) {}
        
        bool operator<(const CircuitWithFitness& other) const {
            return fitness > other.fitness;
        }
    };
    
    std::vector<CircuitWithFitness> top_circuits;
    double best_overall_fitness = 0.0;
    
    std::vector<std::tuple<int, double, int, int>> parameter_sets = {
        {400, 0.25, 100, 50000},
        {150, 0.2, 150, 50000},
        {200, 0.3, 200, 500000}
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
        
        top_circuits.push_back(CircuitWithFitness(best_circuit, best_fitness));
        
        std::sort(top_circuits.begin(), top_circuits.end());
        if (top_circuits.size() > 10) {
            top_circuits.resize(10);
        }
        
        if (best_fitness > best_overall_fitness) {
            best_overall_fitness = best_fitness;
            
            std::cout << "New best fitness found: " << best_fitness << std::endl;
            
            if (best_fitness >= 0.999) {
                std::cout << "Perfect solution found! Stopping search." << std::endl;
                break;
            }
        }
    }
    
    std::cout << "\n=== Top 10 best solutions ===" << std::endl;
    
    for (size_t i = 0; i < top_circuits.size(); i++) {
        std::cout << "Circuit #" << (i+1) << " - Fitness: " << top_circuits[i].fitness << std::endl;
        
        std::string filename = "sbox_circuit_" + std::to_string(i+1) + "_fitness_" + 
                              std::to_string(top_circuits[i].fitness).substr(0, 6) + ".json";
        
        std::ofstream jsonfile(filename);
        jsonfile << "[\n";
        for (size_t j = 0; j < top_circuits[i].circuit.size(); j++) {
            const auto& gate = top_circuits[i].circuit[j];
            jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
            if (gate.gate_type == "ccx") {
                jsonfile << ", " << gate.q3;
            } else {
                jsonfile << ", -1";
            }
            jsonfile << "]";
            if (j < top_circuits[i].circuit.size() - 1) jsonfile << ",";
            jsonfile << "\n";
        }
        jsonfile << "]\n";
        jsonfile.close();
    }
    
    std::cout << "\n=== Testing the best solution (fitness: " << 
        (top_circuits.empty() ? 0.0 : top_circuits[0].fitness) << ") ===" << std::endl;
    
    std::vector<TestVector> final_test_vectors = generate_comprehensive_test_vectors();
    
    if (!top_circuits.empty()) {
        CircuitConfig original_best = top_circuits[0].circuit;
        CircuitConfig simplified_best = simplify_circuit(original_best);
        
        std::cout << "Original circuit had " << original_best.size() << " gates." << std::endl;
        std::cout << "Simplified circuit has " << simplified_best.size() << " gates." << std::endl;
        std::cout << "Gate reduction: " << (original_best.size() - simplified_best.size()) 
                  << " gates (" << std::fixed << std::setprecision(2) 
                  << ((original_best.size() - simplified_best.size()) * 100.0 / original_best.size()) 
                  << "% reduction)" << std::endl;
        
        std::cout << "\n=== Testing the simplified best solution ===" << std::endl;
        test_circuit(simplified_best, final_test_vectors);
        
        auto analyze_circuit = [](const CircuitConfig& circuit, const std::string& label) {
            int x_count = 0;
            int cx_count = 0;
            int ccx_count = 0;
            int swap_count = 0;
            std::map<int, int> target_qubit_counts;
            
            for (const auto& gate : circuit) {
                if (gate.gate_type == "x") {
                    x_count++;
                    target_qubit_counts[gate.q1]++;
                } else if (gate.gate_type == "cx") {
                    cx_count++;
                    target_qubit_counts[gate.q1]++;
                    target_qubit_counts[gate.q2]++;
                } else if (gate.gate_type == "ccx") {
                    ccx_count++;
                    target_qubit_counts[gate.q1]++;
                    target_qubit_counts[gate.q2]++;
                    target_qubit_counts[gate.q3]++;
                } else if (gate.gate_type == "swap") {
                    swap_count++;
                    target_qubit_counts[gate.q1]++;
                    target_qubit_counts[gate.q2]++;
                }
            }
            
            std::cout << "\n" << label << " Circuit Gate Analysis:" << std::endl;
            std::cout << "Total gates: " << circuit.size() << std::endl;
            std::cout << "X gates: " << x_count << std::endl;
            std::cout << "CX gates: " << cx_count << std::endl;
            std::cout << "CCX gates: " << ccx_count << std::endl;
            std::cout << "SWAP gates: " << swap_count << std::endl;
            
            std::cout << "Qubit usage distribution:" << std::endl;
            std::map<std::string, std::vector<int>> qubit_categories;
            for (const auto& [qubit, count] : target_qubit_counts) {
                if (qubit < 8) {
                    qubit_categories["Input qubits (0-7)"].push_back(qubit);
                } else if (qubit < 16) {
                    qubit_categories["Output qubits (8-15)"].push_back(qubit);
                }
            }
            
            for (const auto& [category, qubits] : qubit_categories) {
                std::cout << "  " << category << ": " << qubits.size() << " qubits used" << std::endl;
            }
            
            std::cout << "Circuit depth estimation: " << circuit.size() << " (worst case)" << std::endl;
        };
        
        analyze_circuit(original_best, "Original Best");
        analyze_circuit(simplified_best, "Simplified Best");
        
        auto calculate_basic_gate_count = [](const CircuitConfig& circuit) {
            int basic_gate_count = 0;
            
            for (const auto& gate : circuit) {
                if (gate.gate_type == "x") {
                    basic_gate_count += 1;
                } else if (gate.gate_type == "cx") {
                    basic_gate_count += 1;
                } else if (gate.gate_type == "ccx") {
                    basic_gate_count += 8;
                } else if (gate.gate_type == "swap") {
                    basic_gate_count += 3;
                }
            }
            
            return basic_gate_count;
        };
        
        int original_basic_gates = calculate_basic_gate_count(original_best);
        int simplified_basic_gates = calculate_basic_gate_count(simplified_best);
        
        std::cout << "\nBasic Gate Count Analysis:" << std::endl;
        std::cout << "Original circuit basic gates: " << original_basic_gates << std::endl;
        std::cout << "Simplified circuit basic gates: " << simplified_basic_gates << std::endl;
        std::cout << "Basic gate reduction: " << (original_basic_gates - simplified_basic_gates) 
                  << " gates (" << std::fixed << std::setprecision(2) 
                  << ((original_basic_gates - simplified_basic_gates) * 100.0 / original_basic_gates) 
                  << "% reduction)" << std::endl;
                  
        std::cout << "\nS-box Implementation Analysis:" << std::endl;
        std::cout << "Classical S-box lookup table size: 256 bytes" << std::endl;
        std::cout << "Quantum circuit gate count: " << simplified_best.size() << " gates" << std::endl;
        std::cout << "Estimated quantum circuit size in terms of basic gates: " << simplified_basic_gates << std::endl;
        
        std::ofstream final_jsonfile("final_optimized_sbox_circuit.json");
        final_jsonfile << "[\n";
        for (size_t j = 0; j < simplified_best.size(); j++) {
            const auto& gate = simplified_best[j];
            final_jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
            if (gate.gate_type == "ccx") {
                final_jsonfile << ", " << gate.q3;
            } else {
                final_jsonfile << ", -1";
            }
            final_jsonfile << "]";
            if (j < simplified_best.size() - 1) final_jsonfile << ",";
            final_jsonfile << "\n";
        }
        final_jsonfile << "]\n";
        final_jsonfile.close();
        
        std::cout << "\nFinal optimized circuit saved to 'final_optimized_sbox_circuit.json'" << std::endl;
        
        std::cout << "\nCircuit Structure Visualization:" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        
        const int max_display_gates = 20;
        const int display_gates = std::min(static_cast<int>(simplified_best.size()), max_display_gates);
        
        int max_qubit = 0;
        for (size_t i = 0; i < display_gates; i++) {
            const auto& gate = simplified_best[i];
            if (gate.gate_type == "x") {
                max_qubit = std::max(max_qubit, gate.q1);
            } else if (gate.gate_type == "cx") {
                max_qubit = std::max(max_qubit, gate.q1);
                max_qubit = std::max(max_qubit, gate.q2);
            } else if (gate.gate_type == "ccx") {
                max_qubit = std::max(max_qubit, gate.q1);
                max_qubit = std::max(max_qubit, gate.q2);
                max_qubit = std::max(max_qubit, gate.q3);
            } else if (gate.gate_type == "swap") {
                max_qubit = std::max(max_qubit, gate.q1);
                max_qubit = std::max(max_qubit, gate.q2);
            }
        }
        max_qubit = std::min(max_qubit, 15);
        
        std::cout << "      ";
        for (int i = 0; i < display_gates; i++) {
            std::cout << std::setw(3) << i;
        }
        std::cout << std::endl;
        
        std::cout << "      ";
        for (int i = 0; i < display_gates; i++) {
            std::cout << "---";
        }
        std::cout << std::endl;
        
        for (int q = 0; q <= max_qubit; q++) {
            std::cout << "q" << std::setw(2) << q << " : ";
            
            for (int i = 0; i < display_gates; i++) {
                const auto& gate = simplified_best[i];
                
                if (gate.gate_type == "x" && gate.q1 == q) {
                    std::cout << " X ";
                } else if (gate.gate_type == "cx" && gate.q2 == q) {
                    std::cout << " ⊕ ";
                } else if (gate.gate_type == "cx" && gate.q1 == q) {
                    std::cout << " • ";
                } else if (gate.gate_type == "ccx" && gate.q3 == q) {
                    std::cout << " ⊕ ";
                } else if ((gate.gate_type == "ccx" && (gate.q1 == q || gate.q2 == q))) {
                    std::cout << " • ";
                } else if (gate.gate_type == "swap" && (gate.q1 == q || gate.q2 == q)) {
                    std::cout << " ↔ ";
                } else {
                    std::cout << "───";
                }
            }
            std::cout << std::endl;
        }
        
        if (simplified_best.size() > max_display_gates) {
            std::cout << "(Showing first " << max_display_gates << " gates out of " 
                     << simplified_best.size() << " total gates)" << std::endl;
        }
        
        std::cout << "--------------------------------------------------" << std::endl;
    } else {
        std::cout << "No circuits found!" << std::endl;
    }
    
    std::cout << "\nAES S-box Quantum Implementation Complete!" << std::endl;
    
    return 0;
}