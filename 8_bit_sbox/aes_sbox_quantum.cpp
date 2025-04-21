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
    uint8_t state;
    
public:
    QuantumState(uint8_t initial_state) : state(initial_state) {}
    
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
    
    uint8_t measure() {
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
    
public:
    SBoxQuantumCircuit(uint8_t state, const CircuitConfig& config)
        : input_state(state), circuit_config(config) {}
    
    uint8_t execute_circuit() {
        QuantumState qstate(input_state);
        
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
        
        return qstate.measure();
    }
};

std::vector<TestVector> generate_test_vectors() {
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
    
    return simplified;
}

void test_circuit(const CircuitConfig& circuit, const std::vector<TestVector>& test_vectors) {
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
    }
    
    std::ofstream jsonfile("sbox_circuit_config.json");
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

void print_circuit_gates(const CircuitConfig& circuit) {
    std::cout << "\nCircuit Gates (" << circuit.size() << " total):" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < circuit.size(); i++) {
        const auto& gate = circuit[i];
        std::cout << std::setw(3) << i << ": ";
        
        if (gate.gate_type == "x") {
            std::cout << "X(q" << gate.q1 << ")";
        } else if (gate.gate_type == "cx") {
            std::cout << "CX(q" << gate.q1 << ", q" << gate.q2 << ")";
        } else if (gate.gate_type == "ccx") {
            std::cout << "CCX(q" << gate.q1 << ", q" << gate.q2 << ", q" << gate.q3 << ")";
        } else if (gate.gate_type == "swap") {
            std::cout << "SWAP(q" << gate.q1 << ", q" << gate.q2 << ")";
        }
        
        std::cout << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;
}

void visualize_circuit(const CircuitConfig& circuit) {
    std::cout << "\nCircuit Visualization:" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    
    int max_display_gates = std::min(static_cast<int>(circuit.size()), 30);
    int max_qubit = 7;
    
    std::cout << "      ";
    for (int i = 0; i < max_display_gates; i++) {
        std::cout << std::setw(3) << i;
    }
    std::cout << std::endl;
    
    std::cout << "      ";
    for (int i = 0; i < max_display_gates; i++) {
        std::cout << "---";
    }
    std::cout << std::endl;
    
    for (int q = 0; q <= max_qubit; q++) {
        std::cout << "q" << std::setw(2) << q << " : ";
        
        for (int i = 0; i < max_display_gates; i++) {
            if (i >= circuit.size()) {
                std::cout << "   ";
                continue;
            }
            
            const auto& gate = circuit[i];
            
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
    
    if (circuit.size() > max_display_gates) {
        std::cout << "(Showing first " << max_display_gates << " gates out of " 
                 << circuit.size() << " total gates)" << std::endl;
    }
    
    std::cout << "--------------------------------------------------" << std::endl;
}

void ensure_swap_at_end(CircuitConfig& circuit) {
    std::vector<GateInfo> swap_gates;
    std::vector<GateInfo> non_swap_gates;
    
    for (const auto& gate : circuit) {
        if (gate.gate_type == "swap") {
            swap_gates.push_back(gate);
        } else {
            non_swap_gates.push_back(gate);
        }
    }
    
    circuit.clear();
    circuit.insert(circuit.end(), non_swap_gates.begin(), non_swap_gates.end());
    circuit.insert(circuit.end(), swap_gates.begin(), swap_gates.end());
}

class GeneticAlgorithm {
    private:
        int population_size;
        double mutation_rate;
        int max_gates;
        std::vector<CircuitConfig> population;
        std::mt19937 gen;
        std::vector<TestVector> test_vectors;
        std::atomic<bool> perfect_solution_found;
        int num_threads;
        
        CircuitConfig generate_random_circuit() {
            std::uniform_int_distribution<> num_gates_dist(20, max_gates);
            std::uniform_int_distribution<> qubit_dist(0, 7);
            std::uniform_int_distribution<> gate_type_dist(0, 3);
            
            int num_gates = num_gates_dist(gen);
            CircuitConfig circuit;
            
            for (int i = 0; i < num_gates; i++) {
                int gate_idx = gate_type_dist(gen);
                
                if (gate_idx == 0) {
                    int target = qubit_dist(gen);
                    circuit.push_back(GateInfo("x", target));
                } else if (gate_idx == 1) {
                    int control = qubit_dist(gen);
                    int target = qubit_dist(gen);
                    while (target == control) {
                        target = qubit_dist(gen);
                    }
                    circuit.push_back(GateInfo("cx", control, target));
                } else if (gate_idx == 2) {
                    int control1 = qubit_dist(gen);
                    int control2 = qubit_dist(gen);
                    while (control2 == control1) {
                        control2 = qubit_dist(gen);
                    }
                    int target = qubit_dist(gen);
                    while (target == control1 || target == control2) {
                        target = qubit_dist(gen);
                    }
                    circuit.push_back(GateInfo("ccx", control1, control2, target));
                } else {
                    int q1 = qubit_dist(gen);
                    int q2 = qubit_dist(gen);
                    while (q2 == q1) {
                        q2 = qubit_dist(gen);
                    }
                    circuit.push_back(GateInfo("swap", q1, q2));
                }
            }
            
            ensure_swap_at_end(circuit);
            return circuit;
        }
        
        void mutate_circuit(CircuitConfig& circuit) {
            std::uniform_real_distribution<> prob_dist(0.0, 1.0);
            std::uniform_int_distribution<> gate_type_dist(0, 3);
            std::uniform_int_distribution<> qubit_dist(0, 7);
            
            if (circuit.empty()) {
                circuit = generate_random_circuit();
                return;
            }
            
            std::vector<GateInfo> non_swap_gates;
            std::vector<GateInfo> swap_gates;
            
            for (const auto& gate : circuit) {
                if (gate.gate_type == "swap") {
                    swap_gates.push_back(gate);
                } else {
                    non_swap_gates.push_back(gate);
                }
            }
            
            if (non_swap_gates.empty()) {
                non_swap_gates.push_back(GateInfo("x", qubit_dist(gen)));
            }
            
            std::uniform_int_distribution<> idx_dist(0, non_swap_gates.size() - 1);
            
            if (prob_dist(gen) < 0.3 && non_swap_gates.size() < max_gates) {
                int gate_idx = gate_type_dist(gen);
                
                if (gate_idx == 0) {
                    int target = qubit_dist(gen);
                    non_swap_gates.push_back(GateInfo("x", target));
                } else if (gate_idx == 1) {
                    int control = qubit_dist(gen);
                    int target = qubit_dist(gen);
                    while (target == control) {
                        target = qubit_dist(gen);
                    }
                    non_swap_gates.push_back(GateInfo("cx", control, target));
                } else if (gate_idx == 2) {
                    int control1 = qubit_dist(gen);
                    int control2 = qubit_dist(gen);
                    while (control2 == control1) {
                        control2 = qubit_dist(gen);
                    }
                    int target = qubit_dist(gen);
                    while (target == control1 || target == control2) {
                        target = qubit_dist(gen);
                    }
                    non_swap_gates.push_back(GateInfo("ccx", control1, control2, target));
                } else {
                    int q1 = qubit_dist(gen);
                    int q2 = qubit_dist(gen);
                    while (q2 == q1) {
                        q2 = qubit_dist(gen);
                    }
                    swap_gates.push_back(GateInfo("swap", q1, q2));
                }
            } else if (prob_dist(gen) < 0.3 && non_swap_gates.size() > 5) {
                int idx = idx_dist(gen);
                non_swap_gates.erase(non_swap_gates.begin() + idx);
            } else if (prob_dist(gen) < 0.7 && !non_swap_gates.empty()) {
                int idx = idx_dist(gen);
                int gate_idx = gate_type_dist(gen);
                
                if (gate_idx == 0) {
                    int target = qubit_dist(gen);
                    non_swap_gates[idx] = GateInfo("x", target);
                } else if (gate_idx == 1) {
                    int control = qubit_dist(gen);
                    int target = qubit_dist(gen);
                    while (target == control) {
                        target = qubit_dist(gen);
                    }
                    non_swap_gates[idx] = GateInfo("cx", control, target);
                } else if (gate_idx == 2) {
                    int control1 = qubit_dist(gen);
                    int control2 = qubit_dist(gen);
                    while (control2 == control1) {
                        control2 = qubit_dist(gen);
                    }
                    int target = qubit_dist(gen);
                    while (target == control1 || target == control2) {
                        target = qubit_dist(gen);
                    }
                    non_swap_gates[idx] = GateInfo("ccx", control1, control2, target);
                }
            }
            
            if (prob_dist(gen) < 0.2) {
                if (swap_gates.empty() || prob_dist(gen) < 0.5) {
                    int q1 = qubit_dist(gen);
                    int q2 = qubit_dist(gen);
                    while (q2 == q1) {
                        q2 = qubit_dist(gen);
                    }
                    swap_gates.push_back(GateInfo("swap", q1, q2));
                } else if (!swap_gates.empty() && prob_dist(gen) < 0.5) {
                    std::uniform_int_distribution<> swap_idx_dist(0, swap_gates.size() - 1);
                    int idx = swap_idx_dist(gen);
                    swap_gates.erase(swap_gates.begin() + idx);
                } else if (!swap_gates.empty()) {
                    std::uniform_int_distribution<> swap_idx_dist(0, swap_gates.size() - 1);
                    int idx = swap_idx_dist(gen);
                    int q1 = qubit_dist(gen);
                    int q2 = qubit_dist(gen);
                    while (q2 == q1) {
                        q2 = qubit_dist(gen);
                    }
                    swap_gates[idx] = GateInfo("swap", q1, q2);
                }
            }
            
            circuit.clear();
            circuit.insert(circuit.end(), non_swap_gates.begin(), non_swap_gates.end());
            circuit.insert(circuit.end(), swap_gates.begin(), swap_gates.end());
        }
        
        double evaluate_fitness(const CircuitConfig& circuit) {
            int total_bit_matches = 0;
            int perfect_matches = 0;
            
            for (const auto& test_vec : test_vectors) {
                uint8_t input_state = test_vec.first;
                uint8_t classical_output = test_vec.second;
                
                SBoxQuantumCircuit qc_instance(input_state, circuit);
                uint8_t quantum_output = qc_instance.execute_circuit();
                
                if (quantum_output == classical_output) {
                    perfect_matches++;
                }
                
                for (int bit = 0; bit < 8; bit++) {
                    if (((quantum_output >> bit) & 1) == ((classical_output >> bit) & 1)) {
                        total_bit_matches++;
                    }
                }
            }
            
            double bit_match_rate = static_cast<double>(total_bit_matches) / (test_vectors.size() * 8);
            double perfect_match_rate = static_cast<double>(perfect_matches) / test_vectors.size();
            double size_penalty = std::max(0.0, 1.0 - (circuit.size() / static_cast<double>(max_gates * 2)));
            
            return (0.3 * bit_match_rate + 0.7 * perfect_match_rate) * (0.9 + 0.1 * size_penalty);
        }
        
        void repair_circuit(CircuitConfig& circuit) {
            std::uniform_real_distribution<> prob_dist(0.0, 1.0);
            std::uniform_int_distribution<> qubit_dist(0, 7);
    
            const int sample_size = 16;
            std::uniform_int_distribution<> sample_dist(0, 255);
            std::vector<TestVector> sample_vectors;
            
            for (int i = 0; i < sample_size; i++) {
                uint8_t input = sample_dist(gen);
                sample_vectors.push_back({input, classical_sbox_lookup(input)});
            }
            
            double initial_fitness = 0;
            for (const auto& test_vec : sample_vectors) {
                uint8_t input_state = test_vec.first;
                uint8_t classical_output = test_vec.second;
                
                SBoxQuantumCircuit qc_instance(input_state, circuit);
                uint8_t quantum_output = qc_instance.execute_circuit();
                
                for (int bit = 0; bit < 8; bit++) {
                    if (((quantum_output >> bit) & 1) == ((classical_output >> bit) & 1)) {
                        initial_fitness += 1.0;
                    }
                }
            }
            initial_fitness /= (sample_size * 8);
            
            std::vector<GateInfo> non_swap_gates;
            std::vector<GateInfo> swap_gates;
            
            for (const auto& gate : circuit) {
                if (gate.gate_type == "swap") {
                    swap_gates.push_back(gate);
                } else {
                    non_swap_gates.push_back(gate);
                }
            }
            
            for (int repair_attempts = 0; repair_attempts < 10; repair_attempts++) {
                CircuitConfig temp_circuit = non_swap_gates;
                temp_circuit.insert(temp_circuit.end(), swap_gates.begin(), swap_gates.end());
                
                if (prob_dist(gen) < 0.5) {
                    int gate_pos = std::uniform_int_distribution<>(0, non_swap_gates.size() - 1)(gen);
                    int gate_type = std::uniform_int_distribution<>(0, 2)(gen);
                    
                    if (gate_type == 0) {
                        int target = qubit_dist(gen);
                        temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("x", target));
                    } else if (gate_type == 1) {
                        int control = qubit_dist(gen);
                        int target = qubit_dist(gen);
                        while (target == control) {
                            target = qubit_dist(gen);
                        }
                        temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("cx", control, target));
                    } else {
                        int control1 = qubit_dist(gen);
                        int control2 = qubit_dist(gen);
                        while (control2 == control1) {
                            control2 = qubit_dist(gen);
                        }
                        int target = qubit_dist(gen);
                        while (target == control1 || target == control2) {
                            target = qubit_dist(gen);
                        }
                        temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("ccx", control1, control2, target));
                    }
                } else if (!non_swap_gates.empty()) {
                    int gate_pos = std::uniform_int_distribution<>(0, non_swap_gates.size() - 1)(gen);
                    temp_circuit.erase(temp_circuit.begin() + gate_pos);
                }
                
                double new_fitness = 0;
                for (const auto& test_vec : sample_vectors) {
                    uint8_t input_state = test_vec.first;
                    uint8_t classical_output = test_vec.second;
                    
                    SBoxQuantumCircuit qc_instance(input_state, temp_circuit);
                    uint8_t quantum_output = qc_instance.execute_circuit();
                    
                    for (int bit = 0; bit < 8; bit++) {
                        if (((quantum_output >> bit) & 1) == ((classical_output >> bit) & 1)) {
                            new_fitness += 1.0;
                        }
                    }
                }
                new_fitness /= (sample_size * 8);
                
                if (new_fitness > initial_fitness) {
                    circuit = temp_circuit;
                    initial_fitness = new_fitness;
                }
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
                        fitness_scores[i] = evaluate_fitness(population[i]);
                        
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
        GeneticAlgorithm(int pop_size = 100, double mut_rate = 0.3, int max_g = 200, int threads = 8)
            : population_size(pop_size), mutation_rate(mut_rate), max_gates(max_g),
              gen(std::random_device{}()), perfect_solution_found(false), num_threads(threads) {
            
            test_vectors = generate_test_vectors();
            
            CircuitConfig initial_circuit;
            for (int i = 0; i < 8; i++) {
                initial_circuit.push_back(GateInfo("cx", i, (i + 1) % 8));
            }
            
            for (int i = 0; i < 8; i++) {
                if ((0x63 >> i) & 1) {
                    initial_circuit.push_back(GateInfo("x", i));
                }
            }
            
            for (int i = 0; i < 4; i++) {
                initial_circuit.push_back(GateInfo("swap", i, 7 - i));
            }
            
            population.push_back(initial_circuit);
            
            for (int i = 1; i < population_size; i++) {
                population.push_back(generate_random_circuit());
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
                    std::vector<GateInfo> p1_non_swap;
                    std::vector<GateInfo> p1_swap;
                    std::vector<GateInfo> p2_non_swap;
                    std::vector<GateInfo> p2_swap;
                    
                    for (const auto& gate : parent1) {
                        if (gate.gate_type == "swap") {
                            p1_swap.push_back(gate);
                        } else {
                            p1_non_swap.push_back(gate);
                        }
                    }
                    
                    for (const auto& gate : parent2) {
                        if (gate.gate_type == "swap") {
                            p2_swap.push_back(gate);
                        } else {
                            p2_non_swap.push_back(gate);
                        }
                    }
                    
                    CircuitConfig child;
                    
                    const int test_samples = 8;
                    std::uniform_int_distribution<> input_dist(0, 255);
                    std::vector<TestVector> sample_tests;
                    
                    for (int i = 0; i < test_samples; i++) {
                        uint8_t input = input_dist(gen);
                        sample_tests.push_back({input, classical_sbox_lookup(input)});
                    }
                    
                    if (!p1_non_swap.empty() && !p2_non_swap.empty()) {
                        if (prob_dist(gen) < 0.5) {
                            int chunk_size = 4;
                            if (p1_non_swap.size() > chunk_size && p2_non_swap.size() > chunk_size) {
                                std::vector<double> chunk_fitnesses1(p1_non_swap.size() - chunk_size + 1, 0.0);
                                std::vector<double> chunk_fitnesses2(p2_non_swap.size() - chunk_size + 1, 0.0);
                                
                                for (size_t i = 0; i <= p1_non_swap.size() - chunk_size; i++) {
                                    CircuitConfig chunk(p1_non_swap.begin() + i, p1_non_swap.begin() + i + chunk_size);
                                    for (const auto& test : sample_tests) {
                                        SBoxQuantumCircuit qc(test.first, chunk);
                                        uint8_t result = qc.execute_circuit();
                                        for (int bit = 0; bit < 8; bit++) {
                                            if (((result >> bit) & 1) == ((test.second >> bit) & 1)) {
                                                chunk_fitnesses1[i] += 1.0;
                                            }
                                        }
                                    }
                                    chunk_fitnesses1[i] /= (test_samples * 8);
                                }
                                
                                for (size_t i = 0; i <= p2_non_swap.size() - chunk_size; i++) {
                                    CircuitConfig chunk(p2_non_swap.begin() + i, p2_non_swap.begin() + i + chunk_size);
                                    for (const auto& test : sample_tests) {
                                        SBoxQuantumCircuit qc(test.first, chunk);
                                        uint8_t result = qc.execute_circuit();
                                        for (int bit = 0; bit < 8; bit++) {
                                            if (((result >> bit) & 1) == ((test.second >> bit) & 1)) {
                                                chunk_fitnesses2[i] += 1.0;
                                            }
                                        }
                                    }
                                    chunk_fitnesses2[i] /= (test_samples * 8);
                                }
                                
                                auto max1_it = std::max_element(chunk_fitnesses1.begin(), chunk_fitnesses1.end());
                                auto max2_it = std::max_element(chunk_fitnesses2.begin(), chunk_fitnesses2.end());
                                
                                int best_chunk1 = std::distance(chunk_fitnesses1.begin(), max1_it);
                                int best_chunk2 = std::distance(chunk_fitnesses2.begin(), max2_it);
                                
                                for (size_t i = 0; i < p1_non_swap.size(); i++) {
                                    if (i < best_chunk1 || i >= best_chunk1 + chunk_size) {
                                        child.push_back(p1_non_swap[i]);
                                    }
                                }
                                
                                for (int i = best_chunk2; i < best_chunk2 + chunk_size; i++) {
                                    child.push_back(p2_non_swap[i]);
                                }
                            } else {
                                std::uniform_int_distribution<> crossover_point_dist(1, std::min(p1_non_swap.size(), p2_non_swap.size()) - 1);
                                int crossover_point = crossover_point_dist(gen);
                                
                                for (int i = 0; i < crossover_point && i < p1_non_swap.size(); i++) {
                                    child.push_back(p1_non_swap[i]);
                                }
                                
                                for (int i = crossover_point; i < p2_non_swap.size(); i++) {
                                    child.push_back(p2_non_swap[i]);
                                }
                            }
                        } else {
                            int num_bits = 4;
                            std::bitset<8> better_bits = 0;
                            
                            for (int bit = 0; bit < 8; bit++) {
                                int p1_correct = 0;
                                int p2_correct = 0;
                                
                                for (const auto& test : sample_tests) {
                                    SBoxQuantumCircuit qc1(test.first, p1_non_swap);
                                    SBoxQuantumCircuit qc2(test.first, p2_non_swap);
                                    
                                    uint8_t result1 = qc1.execute_circuit();
                                    uint8_t result2 = qc2.execute_circuit();
                                    
                                    if (((result1 >> bit) & 1) == ((test.second >> bit) & 1)) {
                                        p1_correct++;
                                    }
                                    
                                    if (((result2 >> bit) & 1) == ((test.second >> bit) & 1)) {
                                        p2_correct++;
                                    }
                                }
                                
                                if (p1_correct > p2_correct) {
                                    better_bits[bit] = 1;
                                }
                            }
                            
                            int p1_gates = 0;
                            int p2_gates = 0;
                            
                            for (int i = 0; i < num_bits; i++) {
                                if (better_bits[i]) {
                                    p1_gates++;
                                } else {
                                    p2_gates++;
                                }
                            }
                            
                            if (p1_gates > 0) {
                                for (const auto& gate : p1_non_swap) {
                                    child.push_back(gate);
                                }
                            }
                            
                            if (p2_gates > 0) {
                                for (const auto& gate : p2_non_swap) {
                                    child.push_back(gate);
                                }
                            }
                            
                            if (p1_gates == 0 && p2_gates == 0) {
                                for (const auto& gate : p1_non_swap) {
                                    child.push_back(gate);
                                }
                            }
                        }
                    } else if (!p1_non_swap.empty()) {
                        child = p1_non_swap;
                    } else if (!p2_non_swap.empty()) {
                        child = p2_non_swap;
                    }
                    
                    std::vector<GateInfo> combined_swaps;
                    if (prob_dist(gen) < 0.5) {
                        combined_swaps = p1_swap;
                    } else {
                        combined_swaps = p2_swap;
                    }
                    
                    if (prob_dist(gen) < 0.2) {
                        std::uniform_int_distribution<> qubit_dist(0, 7);
                        int num_extra_swaps = std::uniform_int_distribution<>(0, 3)(gen);
                        
                        for (int i = 0; i < num_extra_swaps; i++) {
                            int q1 = qubit_dist(gen);
                            int q2 = qubit_dist(gen);
                            while (q2 == q1) {
                                q2 = qubit_dist(gen);
                            }
                            combined_swaps.push_back(GateInfo("swap", q1, q2));
                        }
                    }
                    
                    child.insert(child.end(), combined_swaps.begin(), combined_swaps.end());
                    
                    if (child.empty()) {
                        child = generate_random_circuit();
                    }
                    
                    if (prob_dist(gen) < 0.3) {
                        repair_circuit(child);
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
            CircuitConfig best_circuit;
            
            std::cout << "Starting evolutionary optimization..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (int generation = 0; generation < generations; generation++) {
                std::vector<double> fitness_scores = parallel_fitness_evaluation();
                
                auto max_it = std::max_element(fitness_scores.begin(), fitness_scores.end());
                double current_best_fitness = *max_it;
                int best_idx = std::distance(fitness_scores.begin(), max_it);
                
                if (current_best_fitness > best_fitness) {
                    best_fitness = current_best_fitness;
                    best_circuit = population[best_idx];
                    
                    std::cout << "Generation " << (generation + 1) << ": Best Fitness = " 
                              << std::fixed << std::setprecision(4) << best_fitness;
                    
                    if (best_fitness >= 0.999) {
                        std::cout << " - Perfect solution found!" << std::endl;
                        break;
                    }
                    std::cout << " (Gates: " << best_circuit.size() << ")" << std::endl;
                }
                
                if (generation % 10 == 0 && generation > 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                    
                    std::cout << "Generation " << generation << ": Best fitness = " 
                              << std::fixed << std::setprecision(4) << best_fitness
                              << " (Time elapsed: " << elapsed << "s)" << std::endl;
                }
                
                if (perfect_solution_found) {
                    std::cout << "Perfect solution found at generation " << generation << "!" << std::endl;
                    break;
                }
                
                std::vector<CircuitConfig> selected_population = selection(fitness_scores);
                population = crossover(selected_population);
                population = mutation(population);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
            
            std::cout << "Evolution completed in " << duration << " seconds" << std::endl;
            
            return {best_circuit, best_fitness};
        }
    };

int main() {
    std::vector<TestVector> test_vectors = generate_test_vectors();
    
    std::cout << "Running Genetic Algorithm to find optimal S-box circuit..." << std::endl;
    
    GeneticAlgorithm genetic_algo(200, 0.4, 150, 16);
    auto [best_circuit, best_fitness] = genetic_algo.evolve(2000);
    
    ensure_swap_at_end(best_circuit);
    CircuitConfig simplified_best = simplify_circuit(best_circuit);
    
    std::cout << "\nGenetic Algorithm Best Solution:" << std::endl;
    std::cout << "Original gates: " << best_circuit.size() << std::endl;
    std::cout << "Simplified gates: " << simplified_best.size() << std::endl;
    std::cout << "Fitness score: " << best_fitness << std::endl;
    
    std::cout << "\nTesting Genetic Algorithm Solution:" << std::endl;
    test_circuit(simplified_best, test_vectors);
    print_circuit_gates(simplified_best);
    visualize_circuit(simplified_best);
    
    return 0;
}