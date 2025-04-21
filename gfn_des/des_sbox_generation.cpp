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

const std::vector<std::vector<uint8_t>> DES_SBOX1 = {
    {14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7},
    {0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8},
    {4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0},
    {15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13}};

struct GateInfo
{
    std::string gate_type;
    int q1;
    int q2;
    int q3;

    GateInfo(std::string type, int qubit1, int qubit2 = -1, int qubit3 = -1)
        : gate_type(type), q1(qubit1), q2(qubit2), q3(qubit3) {}

    GateInfo(const GateInfo &other)
        : gate_type(other.gate_type), q1(other.q1), q2(other.q2), q3(other.q3) {}
};

using CircuitConfig = std::vector<GateInfo>;
using TestVector = std::pair<uint8_t, uint8_t>;

class QuantumState
{
private:
    uint8_t state;

public:
    QuantumState(uint8_t initial_state) : state(initial_state) {}

    void apply_x(int target)
    {
        state ^= (1 << target);
    }

    void apply_cx(int control, int target)
    {
        if ((state & (1 << control)) != 0)
        {
            state ^= (1 << target);
        }
    }

    void apply_ccx(int control1, int control2, int target)
    {
        if ((state & (1 << control1)) != 0 && (state & (1 << control2)) != 0)
        {
            state ^= (1 << target);
        }
    }

    void apply_swap(int q1, int q2)
    {
        int bit1 = (state >> q1) & 1;
        int bit2 = (state >> q2) & 1;

        if (bit1 != bit2)
        {
            state ^= (1 << q1);
            state ^= (1 << q2);
        }
    }

    uint8_t measure()
    {
        return state;
    }
};

uint8_t classical_sbox_lookup(uint8_t input)
{
    input &= 0x3F;
    uint8_t row = ((input & 0x20) >> 4) | (input & 0x01);
    uint8_t col = (input & 0x1E) >> 1;
    return DES_SBOX1[row][col] & 0x0F;
}

class SBoxQuantumCircuit
{
private:
    uint8_t input_state;
    CircuitConfig circuit_config;

public:
    SBoxQuantumCircuit(uint8_t state, const CircuitConfig &config)
        : input_state(state), circuit_config(config) {}

    uint8_t execute_circuit()
    {
        QuantumState qstate(input_state & 0x3F);

        for (const auto &gate_info : circuit_config)
        {
            if (gate_info.gate_type == "x")
            {
                qstate.apply_x(gate_info.q1);
            }
            else if (gate_info.gate_type == "cx")
            {
                qstate.apply_cx(gate_info.q1, gate_info.q2);
            }
            else if (gate_info.gate_type == "ccx")
            {
                qstate.apply_ccx(gate_info.q1, gate_info.q2, gate_info.q3);
            }
            else if (gate_info.gate_type == "swap")
            {
                qstate.apply_swap(gate_info.q1, gate_info.q2);
            }
        }

        return qstate.measure() & 0x0F;
    }
};

std::vector<TestVector> generate_test_vectors()
{
    std::vector<TestVector> test_vectors;

    for (int i = 0; i < 64; i++)
    {
        uint8_t input = static_cast<uint8_t>(i);
        uint8_t output = classical_sbox_lookup(input);
        test_vectors.push_back({input, output});
    }

    return test_vectors;
}

CircuitConfig simplify_circuit(const CircuitConfig &circuit)
{
    CircuitConfig simplified = circuit;

    for (size_t i = 0; i < simplified.size() - 1; i++)
    {
        if (simplified[i].gate_type == "x" &&
            i + 1 < simplified.size() &&
            simplified[i + 1].gate_type == "x" &&
            simplified[i].q1 == simplified[i + 1].q1)
        {

            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }

    for (size_t i = 0; i < simplified.size() - 1; i++)
    {
        if (simplified[i].gate_type == "cx" &&
            i + 1 < simplified.size() &&
            simplified[i + 1].gate_type == "cx" &&
            simplified[i].q1 == simplified[i + 1].q1 &&
            simplified[i].q2 == simplified[i + 1].q2)
        {

            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }

    for (size_t i = 0; i < simplified.size() - 1; i++)
    {
        if (simplified[i].gate_type == "ccx" &&
            i + 1 < simplified.size() &&
            simplified[i + 1].gate_type == "ccx" &&
            simplified[i].q1 == simplified[i + 1].q1 &&
            simplified[i].q2 == simplified[i + 1].q2 &&
            simplified[i].q3 == simplified[i + 1].q3)
        {

            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }

    for (size_t i = 0; i < simplified.size() - 1; i++)
    {
        if (simplified[i].gate_type == "swap" &&
            i + 1 < simplified.size() &&
            simplified[i + 1].gate_type == "swap" &&
            ((simplified[i].q1 == simplified[i + 1].q1 &&
              simplified[i].q2 == simplified[i + 1].q2) ||
             (simplified[i].q1 == simplified[i + 1].q2 &&
              simplified[i].q2 == simplified[i + 1].q1)))
        {

            simplified.erase(simplified.begin() + i, simplified.begin() + i + 2);
            i--;
        }
    }

    for (size_t i = 0; i < simplified.size(); i++)
    {
        if (simplified[i].gate_type == "swap" &&
            simplified[i].q1 == simplified[i].q2)
        {
            simplified.erase(simplified.begin() + i);
            i--;
        }
    }

    return simplified;
}

void test_circuit(const CircuitConfig &circuit, const std::vector<TestVector> &test_vectors)
{
    int total_tests = test_vectors.size();
    int passed_tests = 0;
    std::vector<std::map<std::string, uint8_t>> failed_tests;

    for (const auto &test_vec : test_vectors)
    {
        uint8_t input_state = test_vec.first & 0x3F;
        uint8_t classical_output = test_vec.second & 0x0F;

        SBoxQuantumCircuit qc_instance(input_state, circuit);
        uint8_t quantum_output = qc_instance.execute_circuit() & 0x0F;

        if (quantum_output == classical_output)
        {
            passed_tests++;
        }
        else
        {
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

    if (!failed_tests.empty() && failed_tests.size() <= 100)
    {
        std::ofstream csvfile("failed_sbox_tests.csv");
        csvfile << "Input (Hex),Classical Output (Hex),Quantum Output (Hex)" << std::endl;

        for (const auto &test : failed_tests)
        {
            csvfile << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("input")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("classical_output")) << ","
                    << "0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(test.at("quantum_output")) << std::endl;
        }

        csvfile.close();
    }

    std::ofstream jsonfile("sbox_circuit_config.json");
    jsonfile << "[\n";
    for (size_t i = 0; i < circuit.size(); i++)
    {
        const auto &gate = circuit[i];
        jsonfile << "  [\"" << gate.gate_type << "\", " << gate.q1 << ", " << gate.q2;
        if (gate.gate_type == "ccx")
        {
            jsonfile << ", " << gate.q3;
        }
        else
        {
            jsonfile << ", -1";
        }
        jsonfile << "]";
        if (i < circuit.size() - 1)
            jsonfile << ",";
        jsonfile << "\n";
    }
    jsonfile << "]\n";
    jsonfile.close();
}

void print_circuit_gates(const CircuitConfig &circuit)
{
    std::cout << "\nCircuit Gates (" << circuit.size() << " total):" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    for (size_t i = 0; i < circuit.size(); i++)
    {
        const auto &gate = circuit[i];
        std::cout << std::setw(3) << i << ": ";

        if (gate.gate_type == "x")
        {
            std::cout << "X(q" << gate.q1 << ")";
        }
        else if (gate.gate_type == "cx")
        {
            std::cout << "CX(q" << gate.q1 << ", q" << gate.q2 << ")";
        }
        else if (gate.gate_type == "ccx")
        {
            std::cout << "CCX(q" << gate.q1 << ", q" << gate.q2 << ", q" << gate.q3 << ")";
        }
        else if (gate.gate_type == "swap")
        {
            std::cout << "SWAP(q" << gate.q1 << ", q" << gate.q2 << ")";
        }

        std::cout << std::endl;
    }
    std::cout << "--------------------------------------------------" << std::endl;
}

void ensure_swap_at_end(CircuitConfig &circuit, int max_swap_depth = 3)
{
    std::vector<GateInfo> swap_gates;
    std::vector<GateInfo> non_swap_gates;

    for (const auto &gate : circuit)
    {
        if (gate.gate_type == "swap")
        {
            // Only include swaps for output qubits (0-3) and limit to max_depth
            if ((gate.q1 < 4 || gate.q2 < 4) && swap_gates.size() < max_swap_depth)
            {
                swap_gates.push_back(gate);
            }
        }
        else
        {
            non_swap_gates.push_back(gate);
        }
    }

    circuit.clear();
    circuit.insert(circuit.end(), non_swap_gates.begin(), non_swap_gates.end());
    circuit.insert(circuit.end(), swap_gates.begin(), swap_gates.end());
}

class ImprovedGeneticAlgorithm
{
private:
    int population_size;
    double mutation_rate;
    int max_gates;
    std::vector<CircuitConfig> population;
    std::mt19937 gen;
    std::vector<TestVector> test_vectors;
    std::atomic<bool> perfect_solution_found;
    int num_threads;
    int max_generations_without_improvement;
    int stagnation_counter;
    double best_fitness_so_far;
    CircuitConfig best_circuit_ever;
    std::vector<std::pair<int, int>> qubit_pairs;
    int max_swap_depth;

    void generate_qubit_combinations()
    {
        // Generate all possible control qubit combinations
        for (int i = 0; i < 6; i++)
        {
            for (int j = i + 1; j < 6; j++)
            {
                qubit_pairs.push_back({i, j});
            }
        }
    }

    CircuitConfig generate_random_circuit()
    {
        std::uniform_int_distribution<> num_gates_dist(50, max_gates);
        std::uniform_int_distribution<> gate_type_dist(0, 3); // X, CX, CCX, SWAP

        int num_gates = num_gates_dist(gen);
        CircuitConfig circuit;

        for (int i = 0; i < num_gates; i++)
        {
            int gate_idx = gate_type_dist(gen);

            if (gate_idx == 0)
            {
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                circuit.push_back(GateInfo("x", target));
            }
            else if (gate_idx == 1)
            {
                int control = std::uniform_int_distribution<>(0, 5)(gen);
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                while (target == control)
                {
                    target = std::uniform_int_distribution<>(0, 5)(gen);
                }
                circuit.push_back(GateInfo("cx", control, target));
            }
            else if (gate_idx == 2)
            {
                auto pair = qubit_pairs[std::uniform_int_distribution<>(0, qubit_pairs.size() - 1)(gen)];
                int control1 = pair.first;
                int control2 = pair.second;
                int target = std::uniform_int_distribution<>(0, 3)(gen);
                circuit.push_back(GateInfo("ccx", control1, control2, target));
            }
            else if (gate_idx == 3)
            {
                // SWAP gate limited to output qubits
                int q1 = std::uniform_int_distribution<>(0, 3)(gen);
                int q2 = std::uniform_int_distribution<>(0, 3)(gen);
                while (q2 == q1)
                {
                    q2 = std::uniform_int_distribution<>(0, 3)(gen);
                }
                circuit.push_back(GateInfo("swap", q1, q2));
            }
        }

        ensure_swap_at_end(circuit, max_swap_depth);
        return circuit;
    }

    void mutate_circuit(CircuitConfig &circuit)
    {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<> gate_type_dist(0, 3); // X, CX, CCX, SWAP

        if (circuit.empty())
        {
            circuit = generate_random_circuit();
            return;
        }

        std::vector<GateInfo> non_swap_gates;
        std::vector<GateInfo> swap_gates;

        for (const auto &gate : circuit)
        {
            if (gate.gate_type == "swap")
            {
                swap_gates.push_back(gate);
            }
            else
            {
                non_swap_gates.push_back(gate);
            }
        }

        if (non_swap_gates.empty())
        {
            non_swap_gates.push_back(GateInfo("x", std::uniform_int_distribution<>(0, 5)(gen)));
        }

        std::uniform_int_distribution<> idx_dist(0, non_swap_gates.size() - 1);

        double add_prob = 0.4;
        double remove_prob = 0.3;
        double modify_prob = 0.7;

        if (circuit.size() > max_gates * 0.9)
        {
            add_prob = 0.1;
            remove_prob = 0.5;
        }
        else if (circuit.size() < max_gates * 0.5)
        {
            add_prob = 0.6;
            remove_prob = 0.1;
        }

        if (prob_dist(gen) < add_prob && non_swap_gates.size() < max_gates)
        {
            int gate_idx = gate_type_dist(gen);

            if (gate_idx == 0)
            {
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                non_swap_gates.push_back(GateInfo("x", target));
            }
            else if (gate_idx == 1)
            {
                int control = std::uniform_int_distribution<>(0, 5)(gen);
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                while (target == control)
                {
                    target = std::uniform_int_distribution<>(0, 5)(gen);
                }
                non_swap_gates.push_back(GateInfo("cx", control, target));
            }
            else if (gate_idx == 2)
            {
                auto pair = qubit_pairs[std::uniform_int_distribution<>(0, qubit_pairs.size() - 1)(gen)];
                int control1 = pair.first;
                int control2 = pair.second;
                int target = std::uniform_int_distribution<>(0, 3)(gen);
                non_swap_gates.push_back(GateInfo("ccx", control1, control2, target));
            }
            else if (gate_idx == 3 && swap_gates.size() < max_swap_depth)
            {
                int q1 = std::uniform_int_distribution<>(0, 3)(gen);
                int q2 = std::uniform_int_distribution<>(0, 3)(gen);
                while (q2 == q1)
                {
                    q2 = std::uniform_int_distribution<>(0, 3)(gen);
                }
                swap_gates.push_back(GateInfo("swap", q1, q2));
            }
        }
        else if (prob_dist(gen) < remove_prob && non_swap_gates.size() > 5)
        {
            int idx = idx_dist(gen);
            non_swap_gates.erase(non_swap_gates.begin() + idx);
        }
        else if (prob_dist(gen) < modify_prob && !non_swap_gates.empty())
        {
            int idx = idx_dist(gen);
            int gate_idx = gate_type_dist(gen);

            if (gate_idx == 0)
            {
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                non_swap_gates[idx] = GateInfo("x", target);
            }
            else if (gate_idx == 1)
            {
                int control = std::uniform_int_distribution<>(0, 5)(gen);
                int target = std::uniform_int_distribution<>(0, 5)(gen);
                while (target == control)
                {
                    target = std::uniform_int_distribution<>(0, 5)(gen);
                }
                non_swap_gates[idx] = GateInfo("cx", control, target);
            }
            else if (gate_idx == 2)
            {
                auto pair = qubit_pairs[std::uniform_int_distribution<>(0, qubit_pairs.size() - 1)(gen)];
                int control1 = pair.first;
                int control2 = pair.second;
                int target = std::uniform_int_distribution<>(0, 3)(gen);
                non_swap_gates[idx] = GateInfo("ccx", control1, control2, target);
            }
        }

        if (prob_dist(gen) < 0.3 && swap_gates.size() < max_swap_depth)
        {
            if (swap_gates.empty() || prob_dist(gen) < 0.5)
            {
                int q1 = std::uniform_int_distribution<>(0, 3)(gen);
                int q2 = std::uniform_int_distribution<>(0, 3)(gen);
                while (q2 == q1)
                {
                    q2 = std::uniform_int_distribution<>(0, 3)(gen);
                }
                swap_gates.push_back(GateInfo("swap", q1, q2));
            }
            else if (!swap_gates.empty() && prob_dist(gen) < 0.5)
            {
                std::uniform_int_distribution<> swap_idx_dist(0, swap_gates.size() - 1);
                int idx = swap_idx_dist(gen);
                swap_gates.erase(swap_gates.begin() + idx);
            }
            else if (!swap_gates.empty())
            {
                std::uniform_int_distribution<> swap_idx_dist(0, swap_gates.size() - 1);
                int idx = swap_idx_dist(gen);
                int q1 = std::uniform_int_distribution<>(0, 3)(gen);
                int q2 = std::uniform_int_distribution<>(0, 3)(gen);
                while (q2 == q1)
                {
                    q2 = std::uniform_int_distribution<>(0, 3)(gen);
                }
                swap_gates[idx] = GateInfo("swap", q1, q2);
            }
        }

        circuit.clear();
        circuit.insert(circuit.end(), non_swap_gates.begin(), non_swap_gates.end());
        circuit.insert(circuit.end(), swap_gates.begin(), swap_gates.end());
    }

    double evaluate_fitness(const CircuitConfig &circuit)
    {
        int total_bit_matches = 0;
        int perfect_matches = 0;

        for (const auto &test_vec : test_vectors)
        {
            uint8_t input_state = test_vec.first & 0x3F;
            uint8_t classical_output = test_vec.second & 0x0F;

            SBoxQuantumCircuit qc_instance(input_state, circuit);
            uint8_t quantum_output = qc_instance.execute_circuit() & 0x0F;

            if (quantum_output == classical_output)
            {
                perfect_matches++;
            }

            for (int bit = 0; bit < 4; bit++)
            {
                if (((quantum_output >> bit) & 1) == ((classical_output >> bit) & 1))
                {
                    total_bit_matches++;
                }
            }
        }

        double bit_match_rate = static_cast<double>(total_bit_matches) / (test_vectors.size() * 4);
        double perfect_match_rate = static_cast<double>(perfect_matches) / test_vectors.size();
        double size_penalty = std::max(0.0, 1.0 - (circuit.size() / static_cast<double>(max_gates * 2)));

        return (0.2 * bit_match_rate + 0.8 * perfect_match_rate) * (0.95 + 0.05 * size_penalty);
    }

    void repair_and_optimize_circuit(CircuitConfig &circuit)
    {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);

        const int sample_size = 32;
        std::uniform_int_distribution<> sample_dist(0, 63);
        std::vector<TestVector> sample_vectors;

        for (int i = 0; i < sample_size; i++)
        {
            uint8_t input = sample_dist(gen);
            sample_vectors.push_back({input, classical_sbox_lookup(input)});
        }

        double initial_fitness = 0;
        for (const auto &test_vec : sample_vectors)
        {
            uint8_t input_state = test_vec.first & 0x3F;
            uint8_t classical_output = test_vec.second & 0x0F;

            SBoxQuantumCircuit qc_instance(input_state, circuit);
            uint8_t quantum_output = qc_instance.execute_circuit() & 0x0F;

            if (quantum_output == classical_output)
            {
                initial_fitness += 1.0;
            }
        }
        initial_fitness /= sample_size;

        std::vector<GateInfo> non_swap_gates;
        std::vector<GateInfo> swap_gates;

        for (const auto &gate : circuit)
        {
            if (gate.gate_type == "swap")
            {
                swap_gates.push_back(gate);
            }
            else
            {
                non_swap_gates.push_back(gate);
            }
        }

        for (int repair_attempts = 0; repair_attempts < 20; repair_attempts++)
        {
            CircuitConfig temp_circuit = non_swap_gates;
            temp_circuit.insert(temp_circuit.end(), swap_gates.begin(), swap_gates.end());

            if (prob_dist(gen) < 0.5)
            {
                int gate_pos = std::uniform_int_distribution<>(0, non_swap_gates.size() - 1)(gen);
                int gate_type = std::uniform_int_distribution<>(0, 2)(gen);

                if (gate_type == 0)
                {
                    int target = std::uniform_int_distribution<>(0, 5)(gen);
                    temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("x", target));
                }
                else if (gate_type == 1)
                {
                    int control = std::uniform_int_distribution<>(0, 5)(gen);
                    int target = std::uniform_int_distribution<>(0, 5)(gen);
                    while (target == control)
                    {
                        target = std::uniform_int_distribution<>(0, 5)(gen);
                    }
                    temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("cx", control, target));
                }
                else if (gate_type == 2)
                {
                    auto pair = qubit_pairs[std::uniform_int_distribution<>(0, qubit_pairs.size() - 1)(gen)];
                    int control1 = pair.first;
                    int control2 = pair.second;
                    int target = std::uniform_int_distribution<>(0, 3)(gen);
                    temp_circuit.insert(temp_circuit.begin() + gate_pos, GateInfo("ccx", control1, control2, target));
                }
            }
            else if (!non_swap_gates.empty())
            {
                int gate_pos = std::uniform_int_distribution<>(0, non_swap_gates.size() - 1)(gen);
                temp_circuit.erase(temp_circuit.begin() + gate_pos);
            }

            double new_fitness = 0;
            for (const auto &test_vec : sample_vectors)
            {
                uint8_t input_state = test_vec.first & 0x3F;
                uint8_t classical_output = test_vec.second & 0x0F;

                SBoxQuantumCircuit qc_instance(input_state, temp_circuit);
                uint8_t quantum_output = qc_instance.execute_circuit() & 0x0F;

                if (quantum_output == classical_output)
                {
                    new_fitness += 1.0;
                }
            }
            new_fitness /= sample_size;

            if (new_fitness > initial_fitness)
            {
                circuit = temp_circuit;
                initial_fitness = new_fitness;
            }
        }
    }

    std::vector<double> parallel_fitness_evaluation()
    {
        std::vector<double> fitness_scores(population.size(), 0.0);
        std::vector<std::future<void>> futures;

        int circuits_per_thread = population.size() / num_threads;
        if (circuits_per_thread == 0)
            circuits_per_thread = 1;

        for (int t = 0; t < num_threads; t++)
        {
            int start_idx = t * circuits_per_thread;
            int end_idx = (t == num_threads - 1) ? population.size() : (t + 1) * circuits_per_thread;

            futures.push_back(std::async(std::launch::async, [this, start_idx, end_idx, &fitness_scores]()
                                         {
                for (int i = start_idx; i < end_idx; i++) {
                    fitness_scores[i] = evaluate_fitness(population[i]);
                    
                    if (fitness_scores[i] >= 0.999) {
                        perfect_solution_found = true;
                    }
                } }));
        }

        for (auto &future : futures)
        {
            future.get();
        }

        return fitness_scores;
    }

public:
    ImprovedGeneticAlgorithm(int pop_size = 150, double mut_rate = 0.3, int max_g = 300, int threads = 8)
        : population_size(pop_size), mutation_rate(mut_rate), max_gates(max_g),
          gen(std::random_device{}()), perfect_solution_found(false), num_threads(threads),
          max_generations_without_improvement(50), stagnation_counter(0), best_fitness_so_far(0.0),
          max_swap_depth(3)
    {

        test_vectors = generate_test_vectors();
        generate_qubit_combinations();

        CircuitConfig initial_circuit;

        // Initialize with a different starting strategy
        for (int i = 0; i < 6; i++)
        {
            if (i < 4)
            {
                initial_circuit.push_back(GateInfo("x", i));
            }
        }

        // Add CX gates connecting input qubits to output qubits
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (i >= 4)
                {
                    initial_circuit.push_back(GateInfo("cx", i, j));
                }
            }
        }

        // Add CCX gates for boolean function implementation
        for (int i = 0; i < 6; i++)
        {
            for (int j = i + 1; j < 6; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    if (i != k && j != k)
                    {
                        initial_circuit.push_back(GateInfo("ccx", i, j, k));
                    }
                }
            }
        }

        population.push_back(initial_circuit);

        // Generate diverse starting population
        for (int i = 1; i < population_size; i++)
        {
            if (i < population_size / 3)
            {
                // First third: random circuits
                population.push_back(generate_random_circuit());
            }
            else if (i < 2 * population_size / 3)
            {
                // Second third: mutated versions of initial circuit
                CircuitConfig mutated = initial_circuit;
                for (int j = 0; j < 5; j++)
                {
                    mutate_circuit(mutated);
                }
                population.push_back(mutated);
            }
            else
            {
                // Final third: specialized circuits for specific output bits
                CircuitConfig specialized;
                int output_bit = (i % 4);

                for (int j = 0; j < 6; j++)
                {
                    specialized.push_back(GateInfo("cx", j, output_bit));
                }

                for (int j = 0; j < 6; j++)
                {
                    for (int k = j + 1; k < 6; k++)
                    {
                        specialized.push_back(GateInfo("ccx", j, k, output_bit));
                    }
                }

                population.push_back(specialized);
            }
        }
    }

    std::vector<CircuitConfig> selection(const std::vector<double> &fitness_scores)
    {
        std::vector<CircuitConfig> selected;

        // Elitism - keep the best individual
        auto max_it = std::max_element(fitness_scores.begin(), fitness_scores.end());
        int best_idx = std::distance(fitness_scores.begin(), max_it);
        selected.push_back(population[best_idx]);

        // Tournament selection
        std::uniform_int_distribution<> idx_dist(0, population_size - 1);
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);

        while (selected.size() < population_size)
        {
            std::vector<int> tournament;
            int tournament_size = 5;

            for (int j = 0; j < tournament_size; j++)
            {
                tournament.push_back(idx_dist(gen));
            }

            int winner = tournament[0];
            for (int j = 1; j < tournament_size; j++)
            {
                if (fitness_scores[tournament[j]] > fitness_scores[winner])
                {
                    winner = tournament[j];
                }
            }

            // Occasionally select a random individual to maintain diversity
            if (prob_dist(gen) < 0.05)
            {
                winner = idx_dist(gen);
            }

            selected.push_back(population[winner]);
        }

        return selected;
    }

    std::vector<CircuitConfig> crossover(const std::vector<CircuitConfig> &selected_population)
    {
        std::vector<CircuitConfig> new_population;
        new_population.push_back(selected_population[0]); // Keep the elite individual

        std::uniform_int_distribution<> parent_dist(0, selected_population.size() - 1);
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);

        while (new_population.size() < population_size)
        {
            int parent1_idx = parent_dist(gen);
            int parent2_idx = parent_dist(gen);

            while (parent2_idx == parent1_idx)
            {
                parent2_idx = parent_dist(gen);
            }

            const CircuitConfig &parent1 = selected_population[parent1_idx];
            const CircuitConfig &parent2 = selected_population[parent2_idx];

            if (prob_dist(gen) < 0.8)
            { // Probability of crossover
                std::vector<GateInfo> p1_non_swap;
                std::vector<GateInfo> p1_swap;
                std::vector<GateInfo> p2_non_swap;
                std::vector<GateInfo> p2_swap;

                for (const auto &gate : parent1)
                {
                    if (gate.gate_type == "swap")
                    {
                        p1_swap.push_back(gate);
                    }
                    else
                    {
                        p1_non_swap.push_back(gate);
                    }
                }

                for (const auto &gate : parent2)
                {
                    if (gate.gate_type == "swap")
                    {
                        p2_swap.push_back(gate);
                    }
                    else
                    {
                        p2_non_swap.push_back(gate);
                    }
                }

                CircuitConfig child;

                // Sample test vectors for fitness evaluation
                const int test_samples = 16;
                std::uniform_int_distribution<> input_dist(0, 63);
                std::vector<TestVector> sample_tests;

                for (int i = 0; i < test_samples; i++)
                {
                    uint8_t input = input_dist(gen);
                    sample_tests.push_back({input, classical_sbox_lookup(input)});
                }

                if (!p1_non_swap.empty() && !p2_non_swap.empty())
                {
                    // Intelligent crossover strategies
                    double crossover_type = prob_dist(gen);

                    if (crossover_type < 0.4)
                    {
                        // Per-bit contribution analysis
                        int num_bits = 4; // 4 output bits
                        std::vector<bool> better_bits(num_bits, false);

                        for (int bit = 0; bit < num_bits; bit++)
                        {
                            int p1_correct = 0;
                            int p2_correct = 0;

                            for (const auto &test : sample_tests)
                            {
                                SBoxQuantumCircuit qc1(test.first, p1_non_swap);
                                SBoxQuantumCircuit qc2(test.first, p2_non_swap);

                                uint8_t result1 = qc1.execute_circuit() & 0x0F;
                                uint8_t result2 = qc2.execute_circuit() & 0x0F;

                                if (((result1 >> bit) & 1) == ((test.second >> bit) & 1))
                                {
                                    p1_correct++;
                                }

                                if (((result2 >> bit) & 1) == ((test.second >> bit) & 1))
                                {
                                    p2_correct++;
                                }
                            }

                            better_bits[bit] = (p1_correct > p2_correct);
                        }

                        // Build a circuit focusing on output bits where each parent excels
                        for (const auto &gate : p1_non_swap)
                        {
                            if (gate.gate_type == "x" || gate.gate_type == "cx" || gate.gate_type == "ccx")
                            {
                                if (gate.gate_type == "x")
                                {
                                    child.push_back(gate);
                                }
                                else if (gate.gate_type == "cx" && gate.q2 < 4)
                                {
                                    if (better_bits[gate.q2])
                                    {
                                        child.push_back(gate);
                                    }
                                }
                                else if (gate.gate_type == "ccx" && gate.q3 < 4)
                                {
                                    if (better_bits[gate.q3])
                                    {
                                        child.push_back(gate);
                                    }
                                }
                                else
                                {
                                    child.push_back(gate);
                                }
                            }
                        }

                        for (const auto &gate : p2_non_swap)
                        {
                            if (gate.gate_type == "cx" && gate.q2 < 4)
                            {
                                if (!better_bits[gate.q2])
                                {
                                    child.push_back(gate);
                                }
                            }
                            else if (gate.gate_type == "ccx" && gate.q3 < 4)
                            {
                                if (!better_bits[gate.q3])
                                {
                                    child.push_back(gate);
                                }
                            }
                        }
                    }
                    else if (crossover_type < 0.7)
                    {
                        // Intelligent chunk selection
                        int chunk_size = std::min(8, std::min(static_cast<int>(p1_non_swap.size()), static_cast<int>(p2_non_swap.size())) / 2);

                        if (p1_non_swap.size() > chunk_size && p2_non_swap.size() > chunk_size)
                        {
                            std::vector<double> chunk_fitnesses1(p1_non_swap.size() - chunk_size + 1, 0.0);
                            std::vector<double> chunk_fitnesses2(p2_non_swap.size() - chunk_size + 1, 0.0);

                            // Find the best performing chunks in both parents
                            for (size_t i = 0; i <= p1_non_swap.size() - chunk_size; i++)
                            {
                                CircuitConfig chunk(p1_non_swap.begin() + i, p1_non_swap.begin() + i + chunk_size);
                                for (const auto &test : sample_tests)
                                {
                                    SBoxQuantumCircuit qc(test.first, chunk);
                                    uint8_t result = qc.execute_circuit() & 0x0F;
                                    for (int bit = 0; bit < 4; bit++)
                                    {
                                        if (((result >> bit) & 1) == ((test.second >> bit) & 1))
                                        {
                                            chunk_fitnesses1[i] += 1.0;
                                        }
                                    }
                                }
                                chunk_fitnesses1[i] /= (test_samples * 4);
                            }

                            for (size_t i = 0; i <= p2_non_swap.size() - chunk_size; i++)
                            {
                                CircuitConfig chunk(p2_non_swap.begin() + i, p2_non_swap.begin() + i + chunk_size);
                                for (const auto &test : sample_tests)
                                {
                                    SBoxQuantumCircuit qc(test.first, chunk);
                                    uint8_t result = qc.execute_circuit() & 0x0F;
                                    for (int bit = 0; bit < 4; bit++)
                                    {
                                        if (((result >> bit) & 1) == ((test.second >> bit) & 1))
                                        {
                                            chunk_fitnesses2[i] += 1.0;
                                        }
                                    }
                                }
                                chunk_fitnesses2[i] /= (test_samples * 4);
                            }

                            auto max1_it = std::max_element(chunk_fitnesses1.begin(), chunk_fitnesses1.end());
                            auto max2_it = std::max_element(chunk_fitnesses2.begin(), chunk_fitnesses2.end());

                            int best_chunk1 = std::distance(chunk_fitnesses1.begin(), max1_it);
                            int best_chunk2 = std::distance(chunk_fitnesses2.begin(), max2_it);

                            // Combine the base circuit from parent1 with best chunks from both parents
                            for (size_t i = 0; i < p1_non_swap.size(); i++)
                            {
                                if (i < best_chunk1 || i >= best_chunk1 + chunk_size)
                                {
                                    child.push_back(p1_non_swap[i]);
                                }
                            }

                            for (int i = best_chunk2; i < best_chunk2 + chunk_size && i < p2_non_swap.size(); i++)
                            {
                                child.push_back(p2_non_swap[i]);
                            }
                        }
                        else
                        {
                            // Simple one-point crossover if circuits are too small
                            std::uniform_int_distribution<> crossover_point_dist(1, std::min(p1_non_swap.size(), p2_non_swap.size()) - 1);
                            int crossover_point = crossover_point_dist(gen);

                            for (int i = 0; i < crossover_point && i < p1_non_swap.size(); i++)
                            {
                                child.push_back(p1_non_swap[i]);
                            }

                            for (size_t i = crossover_point; i < p2_non_swap.size(); i++)
                            {
                                child.push_back(p2_non_swap[i]);
                            }
                        }
                    }
                    else
                    {
                        // Alternating gate selection
                        size_t max_len = std::max(p1_non_swap.size(), p2_non_swap.size());
                        for (size_t i = 0; i < max_len; i++)
                        {
                            if (i % 2 == 0 && i < p1_non_swap.size())
                            {
                                child.push_back(p1_non_swap[i]);
                            }
                            else if (i % 2 == 1 && i < p2_non_swap.size())
                            {
                                child.push_back(p2_non_swap[i]);
                            }
                        }
                    }
                }
                else if (!p1_non_swap.empty())
                {
                    child = p1_non_swap;
                }
                else if (!p2_non_swap.empty())
                {
                    child = p2_non_swap;
                }

                // Handle SWAP gates - restricted to output qubits and max depth
                std::vector<GateInfo> combined_swaps;
                std::vector<GateInfo> filtered_p1_swaps;
                std::vector<GateInfo> filtered_p2_swaps;

                for (const auto &gate : p1_swap)
                {
                    if (gate.q1 < 4 && gate.q2 < 4 && filtered_p1_swaps.size() < max_swap_depth)
                    {
                        filtered_p1_swaps.push_back(gate);
                    }
                }

                for (const auto &gate : p2_swap)
                {
                    if (gate.q1 < 4 && gate.q2 < 4 && filtered_p2_swaps.size() < max_swap_depth)
                    {
                        filtered_p2_swaps.push_back(gate);
                    }
                }

                if (prob_dist(gen) < 0.5)
                {
                    combined_swaps = filtered_p1_swaps;
                }
                else
                {
                    combined_swaps = filtered_p2_swaps;
                }

                if (prob_dist(gen) < 0.2 && combined_swaps.size() < max_swap_depth)
                {
                    std::uniform_int_distribution<> output_qubit_dist(0, 3);
                    int num_extra_swaps = std::min(max_swap_depth - combined_swaps.size(),
                                                   (size_t)std::uniform_int_distribution<>(0, 2)(gen));

                    for (int i = 0; i < num_extra_swaps; i++)
                    {
                        int q1 = output_qubit_dist(gen);
                        int q2 = output_qubit_dist(gen);
                        while (q2 == q1)
                        {
                            q2 = output_qubit_dist(gen);
                        }
                        combined_swaps.push_back(GateInfo("swap", q1, q2));
                    }
                }

                child.insert(child.end(), combined_swaps.begin(), combined_swaps.end());

                if (child.empty())
                {
                    child = generate_random_circuit();
                }

                if (prob_dist(gen) < 0.3)
                {
                    repair_and_optimize_circuit(child);
                }

                new_population.push_back(child);
            }
            else
            {
                // No crossover, just select one parent
                new_population.push_back(prob_dist(gen) < 0.5 ? parent1 : parent2);
            }
        }

        return new_population;
    }

    std::vector<CircuitConfig> mutation(std::vector<CircuitConfig> population)
    {
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);

        // Don't mutate the elite individual
        for (int i = 1; i < population.size(); i++)
        {
            if (prob_dist(gen) < mutation_rate)
            {
                mutate_circuit(population[i]);
            }
        }

        return population;
    }

    void reset_population_partially()
    {
        std::cout << "Resetting 70% of population due to stagnation..." << std::endl;

        // Keep top 30%
        std::vector<double> fitness_scores = parallel_fitness_evaluation();
        std::vector<std::pair<double, int>> scored_indices;

        for (int i = 0; i < population_size; i++)
        {
            scored_indices.push_back({fitness_scores[i], i});
        }

        std::sort(scored_indices.begin(), scored_indices.end(), std::greater<>());

        std::vector<CircuitConfig> new_population;
        int keep_count = population_size * 0.3;

        for (int i = 0; i < keep_count; i++)
        {
            new_population.push_back(population[scored_indices[i].second]);
        }

        // Generate new individuals for the rest
        while (new_population.size() < population_size)
        {
            new_population.push_back(generate_random_circuit());
        }

        population = new_population;
    }

    std::pair<CircuitConfig, double> evolve(int generations = 100)
    {
        double best_fitness = 0.0;
        CircuitConfig best_circuit;

        std::cout << "Starting evolutionary optimization with improved algorithm..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int generation = 0; generation < generations; generation++)
        {
            std::vector<double> fitness_scores = parallel_fitness_evaluation();

            auto max_it = std::max_element(fitness_scores.begin(), fitness_scores.end());
            double current_best_fitness = *max_it;
            int best_idx = std::distance(fitness_scores.begin(), max_it);

            if (current_best_fitness > best_fitness_so_far)
            {
                best_fitness_so_far = current_best_fitness;
                best_circuit_ever = population[best_idx];
                stagnation_counter = 0;

                std::cout << "Generation " << (generation + 1) << ": New Best Fitness = "
                          << std::fixed << std::setprecision(4) << best_fitness_so_far;

                if (best_fitness_so_far >= 0.999)
                {
                    std::cout << " - Perfect solution found!" << std::endl;
                    break;
                }
                std::cout << " (Gates: " << best_circuit_ever.size() << ")" << std::endl;
            }
            else
            {
                stagnation_counter++;

                if (stagnation_counter >= max_generations_without_improvement)
                {
                    reset_population_partially();
                    stagnation_counter = 0;

                    // Insert the best circuit ever back into the population
                    population[0] = best_circuit_ever;
                }
            }

            if (current_best_fitness > best_fitness)
            {
                best_fitness = current_best_fitness;
                best_circuit = population[best_idx];
            }

            if (generation % 10 == 0 && generation > 0)
            {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

                std::cout << "Generation " << generation << ": Current best = "
                          << std::fixed << std::setprecision(4) << best_fitness
                          << " (Time elapsed: " << elapsed << "s)" << std::endl;
            }

            if (perfect_solution_found)
            {
                std::cout << "Perfect solution found at generation " << generation << "!" << std::endl;
                break;
            }

            // Adaptive mutation rate based on diversity
            double avg_fitness = 0.0;
            for (const auto &score : fitness_scores)
            {
                avg_fitness += score;
            }
            avg_fitness /= fitness_scores.size();

            double fitness_variance = 0.0;
            for (const auto &score : fitness_scores)
            {
                fitness_variance += (score - avg_fitness) * (score - avg_fitness);
            }
            fitness_variance /= fitness_scores.size();

            if (fitness_variance < 0.01)
            {
                mutation_rate = std::min(0.8, mutation_rate * 1.1);
            }
            else if (fitness_variance > 0.05)
            {
                mutation_rate = std::max(0.1, mutation_rate * 0.9);
            }

            std::vector<CircuitConfig> selected_population = selection(fitness_scores);
            population = crossover(selected_population);
            population = mutation(population);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "Evolution completed in " << duration << " seconds" << std::endl;

        return {best_circuit_ever, best_fitness_so_far};
    }
};

int main()
{
    std::vector<TestVector> test_vectors = generate_test_vectors();

    std::cout << "Running Improved Genetic Algorithm to find optimal DES S-box1 circuit..." << std::endl;

    ImprovedGeneticAlgorithm genetic_algo(500, 0.15, 300, 32);
    auto [best_circuit, best_fitness] = genetic_algo.evolve(100000);

    ensure_swap_at_end(best_circuit, 3);
    CircuitConfig simplified_best = simplify_circuit(best_circuit);

    std::cout << "\nImproved Genetic Algorithm Best Solution:" << std::endl;
    std::cout << "Original gates: " << best_circuit.size() << std::endl;
    std::cout << "Simplified gates: " << simplified_best.size() << std::endl;
    std::cout << "Fitness score: " << best_fitness << std::endl;

    std::cout << "\nTesting Improved Genetic Algorithm Solution:" << std::endl;
    test_circuit(simplified_best, test_vectors);
    print_circuit_gates(simplified_best);

    return 0;
}