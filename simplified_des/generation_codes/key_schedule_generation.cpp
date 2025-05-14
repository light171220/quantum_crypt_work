#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

typedef struct {
    int control;
    int target;
} CXGate;

typedef struct {
    std::vector<CXGate> gates;
    double fitness;
} Circuit;

typedef struct {
    std::bitset<10> key;
    std::bitset<8> target;
    std::bitset<8> expected;
    int round;
} TestCase;

const int POPULATION_SIZE = 200;
const int MAX_GENERATIONS = 1000;
const double MUTATION_RATE = 0.25;
const double CROSSOVER_RATE = 0.85;
const int TOURNAMENT_SIZE = 7;
const int MIN_CIRCUIT_SIZE = 8;
const int MAX_CIRCUIT_SIZE = 12;
const int ELITE_SIZE = 30;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::uniform_int_distribution<int> control_dist(0, 9);
std::uniform_int_distribution<int> target_dist(0, 7);

std::vector<TestCase> generateTestCases();
std::vector<Circuit> initializePopulation();
double evaluateFitness(const Circuit &circuit, const std::vector<TestCase> &testCases);
std::bitset<8> simulateCircuit(const Circuit &circuit, const std::bitset<10> &key, const std::bitset<8> &target, int round);
std::vector<Circuit> selection(const std::vector<Circuit> &population);
Circuit crossover(const Circuit &parent1, const Circuit &parent2);
void mutate(Circuit &circuit);
void printCircuit(const Circuit &circuit);
std::unordered_map<int, int> getP10ToP8Mapping(int round);
std::vector<TestCase> generateAllKeyTestCases(int round);
bool hasRepeatedGates(const Circuit &circuit);
bool isSameGate(const CXGate &gate1, const CXGate &gate2);
std::bitset<8> computeExpected(const std::bitset<10> &key, const std::bitset<8> &target, int round);

std::bitset<8> applyCXGate(const std::bitset<10> &key, const std::bitset<8> &target, int control, int target_bit) {
    std::bitset<8> result = target;
    if (key[control]) {
        result.flip(target_bit);
    }
    return result;
}

int main() {
    std::cout << "Genetic Algorithm for S-DES Key Schedule Optimization\n";
    std::cout << "-----------------------------------------------------\n";
    
    std::vector<TestCase> testCases = generateTestCases();
    std::cout << "Generated " << testCases.size() << " initial test cases.\n";
    
    std::vector<Circuit> population = initializePopulation();
    std::cout << "Initialized population with " << population.size() << " circuits.\n";
    
    for (auto &circuit : population) {
        circuit.fitness = evaluateFitness(circuit, testCases);
    }
    
    std::sort(population.begin(), population.end(), 
              [](const Circuit &a, const Circuit &b) { return a.fitness > b.fitness; });
    
    std::cout << "Initial best fitness: " << population[0].fitness << "\n";
    
    for (int generation = 0; generation < MAX_GENERATIONS; generation++) {
        std::vector<Circuit> newPopulation;
        
        for (int i = 0; i < ELITE_SIZE && i < population.size(); i++) {
            newPopulation.push_back(population[i]);
        }
        
        while (newPopulation.size() < POPULATION_SIZE) {
            std::vector<Circuit> parents = selection(population);
            
            Circuit offspring;
            if (uniform_dist(rng) < CROSSOVER_RATE) {
                offspring = crossover(parents[0], parents[1]);
            } else {
                offspring = parents[0];
            }
            
            mutate(offspring);
            
            if (!hasRepeatedGates(offspring)) {
                offspring.fitness = evaluateFitness(offspring, testCases);
                newPopulation.push_back(offspring);
            }
        }
        
        population = newPopulation;
        
        std::sort(population.begin(), population.end(), 
                  [](const Circuit &a, const Circuit &b) { return a.fitness > b.fitness; });
        
        if (generation % 10 == 0 || generation == MAX_GENERATIONS - 1) {
            std::cout << "Generation " << generation << ": Best fitness = " 
                      << std::fixed << std::setprecision(6) << population[0].fitness 
                      << ", Circuit size = " << population[0].gates.size() << "\n";
        }
        
        if (population[0].fitness >= 0.9999) {
            std::cout << "Perfect solution found at generation " << generation << "!\n";
            break;
        }
    }
    
    std::cout << "\nBest solution (fitness = " << population[0].fitness << "):\n";
    printCircuit(population[0]);
    
    std::cout << "\nVerifying solution with comprehensive test cases...\n";
    
    std::vector<TestCase> allKeyTestCasesRound1 = generateAllKeyTestCases(1);
    std::vector<TestCase> allKeyTestCasesRound2 = generateAllKeyTestCases(2);
    
    int correctRound1 = 0;
    for (const auto &testCase : allKeyTestCasesRound1) {
        std::bitset<8> result = simulateCircuit(population[0], testCase.key, testCase.target, testCase.round);
        if (result == testCase.expected) {
            correctRound1++;
        } else {
            std::cout << "Failed R1: Key=" << testCase.key << ", Target=" << testCase.target 
                      << ", Expected=" << testCase.expected << ", Got=" << result << "\n";
            if (correctRound1 + 5 >= allKeyTestCasesRound1.size()) break;
        }
    }
    
    int correctRound2 = 0;
    for (const auto &testCase : allKeyTestCasesRound2) {
        std::bitset<8> result = simulateCircuit(population[0], testCase.key, testCase.target, testCase.round);
        if (result == testCase.expected) {
            correctRound2++;
        } else {
            std::cout << "Failed R2: Key=" << testCase.key << ", Target=" << testCase.target 
                      << ", Expected=" << testCase.expected << ", Got=" << result << "\n";
            if (correctRound2 + 5 >= allKeyTestCasesRound2.size()) break;
        }
    }
    
    std::cout << "Round 1: Passed " << correctRound1 << " out of " << allKeyTestCasesRound1.size() << " test cases.\n";
    std::cout << "Round 2: Passed " << correctRound2 << " out of " << allKeyTestCasesRound2.size() << " test cases.\n";
    
    if (correctRound1 == allKeyTestCasesRound1.size() && correctRound2 == allKeyTestCasesRound2.size()) {
        std::cout << "PERFECT SOLUTION VERIFIED!\n";
    }
    
    std::cout << "\nGenerated C++ code for the solution:\n";
    std::cout << "-------------------------------------\n";
    std::cout << "std::bitset<8> key_schedule(const std::bitset<10>& key, const std::bitset<8>& target, int round) {\n";
    std::cout << "    std::bitset<8> result = target;\n";
    
    for (const auto &gate : population[0].gates) {
        std::cout << "    if (key[" << gate.control << "]) result.flip(" << gate.target << ");\n";
    }
    
    std::cout << "    return result;\n";
    std::cout << "}\n";
    
    std::cout << "\nGenerated quantum circuit code for the solution:\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "def key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num):\n";
    
    for (const auto &gate : population[0].gates) {
        std::cout << "    circuit.cx(key_qubits[" << gate.control << "], target_qubits[" << gate.target << "])\n";
    }
    
    std::cout << "    return circuit\n";
    
    std::cout << "\nGenerated inverse key_schedule code for the solution:\n";
    std::cout << "----------------------------------------------\n";
    std::cout << "def inverse_key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num):\n";
    
    for (int i = population[0].gates.size() - 1; i >= 0; i--) {
        const auto &gate = population[0].gates[i];
        std::cout << "    circuit.cx(key_qubits[" << gate.control << "], target_qubits[" << gate.target << "])\n";
    }
    
    std::cout << "    return circuit\n";
    
    return 0;
}

bool isSameGate(const CXGate &gate1, const CXGate &gate2) {
    return (gate1.control == gate2.control && gate1.target == gate2.target);
}

bool hasRepeatedGates(const Circuit &circuit) {
    for (size_t i = 0; i < circuit.gates.size(); i++) {
        for (size_t j = i + 1; j < circuit.gates.size(); j++) {
            if (isSameGate(circuit.gates[i], circuit.gates[j])) {
                return true;
            }
        }
    }
    return false;
}

std::vector<TestCase> generateTestCases() {
    std::vector<TestCase> testCases;
    
    const std::vector<int> expansion_indices = {3, 0, 1, 2, 1, 2, 3, 0};
    
    for (int round = 1; round <= 2; round++) {
        auto p10_to_p8_mapping = getP10ToP8Mapping(round);
        
        for (int i = 0; i < 300; i++) {
            TestCase testCase;
            
            std::bitset<10> key;
            for (int j = 0; j < 10; j++) {
                key[j] = uniform_dist(rng) < 0.5;
            }
            testCase.key = key;
            
            std::bitset<8> target;
            for (int j = 0; j < 8; j++) {
                target[j] = uniform_dist(rng) < 0.5;
            }
            testCase.target = target;
            
            testCase.expected = computeExpected(key, target, round);
            testCase.round = round;
            
            testCases.push_back(testCase);
        }
        
        if (round == 1) {
            TestCase testCase;
            testCase.key = std::bitset<10>("1010101010");
            testCase.target = std::bitset<8>("00000000");
            testCase.expected = std::bitset<8>("10110100");
            testCase.round = round;
            testCases.push_back(testCase);
        }
        else {
            TestCase testCase;
            testCase.key = std::bitset<10>("1010101010");
            testCase.target = std::bitset<8>("00000000");
            testCase.expected = std::bitset<8>("01010110");
            testCase.round = round;
            testCases.push_back(testCase);
        }
    }
    
    return testCases;
}

std::bitset<8> computeExpected(const std::bitset<10> &key, const std::bitset<8> &target, int round) {
    const std::vector<int> expansion_indices = {3, 0, 1, 2, 1, 2, 3, 0};
    auto p10_to_p8_mapping = getP10ToP8Mapping(round);
    
    std::bitset<8> expected = target;
    for (int j = 0; j < 8; j++) {
        int key_idx = p10_to_p8_mapping[j];
        int target_idx = expansion_indices[j];
        if (key[key_idx]) {
            expected.flip(target_idx);
        }
    }
    
    return expected;
}

std::vector<TestCase> generateAllKeyTestCases(int round) {
    std::vector<TestCase> testCases;
    testCases.reserve(1024);
    
    std::bitset<8> target("00000000");
    
    for (int keyVal = 0; keyVal < 1024; keyVal++) {
        TestCase testCase;
        testCase.key = std::bitset<10>(keyVal);
        testCase.target = target;
        testCase.round = round;
        testCase.expected = computeExpected(testCase.key, target, round);
        testCases.push_back(testCase);
    }
    
    return testCases;
}

std::unordered_map<int, int> getP10ToP8Mapping(int round) {
    std::unordered_map<int, int> mapping;
    std::vector<int> p10_to_p8_table = {0, 1, 3, 4, 5, 7, 8, 9};
    int shift = (round == 1) ? 1 : 3;
    
    for (int i = 0; i < 8; i++) {
        mapping[i] = (p10_to_p8_table[i] + shift) % 10;
    }
    
    return mapping;
}

std::vector<Circuit> initializePopulation() {
    std::vector<Circuit> population;
    
    for (int i = 0; i < POPULATION_SIZE; i++) {
        Circuit circuit;
        
        std::uniform_int_distribution<int> size_dist(MIN_CIRCUIT_SIZE, MAX_CIRCUIT_SIZE);
        int circuitSize = size_dist(rng);
        
        std::unordered_set<std::string> uniqueGates;
        while (circuit.gates.size() < circuitSize) {
            CXGate gate;
            gate.control = control_dist(rng);
            gate.target = target_dist(rng);
            
            std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
            
            if (uniqueGates.find(gateStr) == uniqueGates.end()) {
                uniqueGates.insert(gateStr);
                circuit.gates.push_back(gate);
            }
        }
        
        population.push_back(circuit);
    }
    
    return population;
}

double evaluateFitness(const Circuit &circuit, const std::vector<TestCase> &testCases) {
    int correctResults = 0;
    
    for (const auto &testCase : testCases) {
        std::bitset<8> result = simulateCircuit(circuit, testCase.key, testCase.target, testCase.round);
        if (result == testCase.expected) {
            correctResults++;
        }
    }
    
    double primaryFitness = static_cast<double>(correctResults) / testCases.size();
    double lengthPenalty = 1.0 - (0.001 * std::max(0, (int)circuit.gates.size() - MIN_CIRCUIT_SIZE));
    
    return primaryFitness * lengthPenalty;
}

std::bitset<8> simulateCircuit(const Circuit &circuit, const std::bitset<10> &key, const std::bitset<8> &target, int round) {
    std::bitset<8> current = target;
    
    for (const auto &gate : circuit.gates) {
        current = applyCXGate(key, current, gate.control, gate.target);
    }
    
    return current;
}

std::vector<Circuit> selection(const std::vector<Circuit> &population) {
    std::vector<Circuit> selected;
    
    for (int i = 0; i < 2; i++) {
        std::vector<int> tournamentIndices;
        for (int j = 0; j < TOURNAMENT_SIZE; j++) {
            std::uniform_int_distribution<int> index_dist(0, population.size() - 1);
            tournamentIndices.push_back(index_dist(rng));
        }
        
        int bestIndex = tournamentIndices[0];
        for (int j = 1; j < TOURNAMENT_SIZE; j++) {
            if (population[tournamentIndices[j]].fitness > population[bestIndex].fitness) {
                bestIndex = tournamentIndices[j];
            }
        }
        
        selected.push_back(population[bestIndex]);
    }
    
    return selected;
}

Circuit crossover(const Circuit &parent1, const Circuit &parent2) {
    Circuit child;
    
    std::uniform_int_distribution<int> p1_dist(0, parent1.gates.size());
    std::uniform_int_distribution<int> p2_dist(0, parent2.gates.size());
    
    int p1_point = p1_dist(rng);
    int p2_point = p2_dist(rng);
    
    std::unordered_set<std::string> uniqueGates;
    
    for (int i = 0; i < p1_point && i < parent1.gates.size(); i++) {
        const auto &gate = parent1.gates[i];
        std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
        
        if (uniqueGates.find(gateStr) == uniqueGates.end()) {
            uniqueGates.insert(gateStr);
            child.gates.push_back(gate);
        }
    }
    
    for (int i = p2_point; i < parent2.gates.size(); i++) {
        const auto &gate = parent2.gates[i];
        std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
        
        if (uniqueGates.find(gateStr) == uniqueGates.end()) {
            uniqueGates.insert(gateStr);
            child.gates.push_back(gate);
        }
    }
    
    while (child.gates.size() < MIN_CIRCUIT_SIZE) {
        bool added = false;
        
        if (p1_point < parent1.gates.size()) {
            const auto &gate = parent1.gates[p1_point];
            std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
            
            if (uniqueGates.find(gateStr) == uniqueGates.end()) {
                uniqueGates.insert(gateStr);
                child.gates.push_back(gate);
                p1_point++;
                added = true;
            }
        }
        
        if (!added && p2_point > 0) {
            p2_point--;
            const auto &gate = parent2.gates[p2_point];
            std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
            
            if (uniqueGates.find(gateStr) == uniqueGates.end()) {
                uniqueGates.insert(gateStr);
                child.gates.push_back(gate);
                added = true;
            }
        }
        
        if (!added) {
            CXGate gate;
            gate.control = control_dist(rng);
            gate.target = target_dist(rng);
            std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
            
            if (uniqueGates.find(gateStr) == uniqueGates.end()) {
                uniqueGates.insert(gateStr);
                child.gates.push_back(gate);
            }
        }
    }
    
    if (child.gates.size() > MAX_CIRCUIT_SIZE) {
        child.gates.resize(MAX_CIRCUIT_SIZE);
    }
    
    return child;
}

void mutate(Circuit &circuit) {
    std::unordered_set<std::string> uniqueGates;
    
    for (const auto &gate : circuit.gates) {
        std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
        uniqueGates.insert(gateStr);
    }
    
    for (size_t i = 0; i < circuit.gates.size(); i++) {
        if (uniform_dist(rng) < MUTATION_RATE) {
            std::string oldGateStr = std::to_string(circuit.gates[i].control) + "-" + 
                                    std::to_string(circuit.gates[i].target);
            uniqueGates.erase(oldGateStr);
            
            int newControl = circuit.gates[i].control;
            int newTarget = circuit.gates[i].target;
            
            if (uniform_dist(rng) < 0.5) {
                newControl = control_dist(rng);
            } else {
                newTarget = target_dist(rng);
            }
            
            std::string newGateStr = std::to_string(newControl) + "-" + std::to_string(newTarget);
            
            if (uniqueGates.find(newGateStr) == uniqueGates.end()) {
                circuit.gates[i].control = newControl;
                circuit.gates[i].target = newTarget;
                uniqueGates.insert(newGateStr);
            } else {
                uniqueGates.insert(oldGateStr);
            }
        }
    }
    
    if (uniform_dist(rng) < MUTATION_RATE && circuit.gates.size() < MAX_CIRCUIT_SIZE) {
        CXGate gate;
        gate.control = control_dist(rng);
        gate.target = target_dist(rng);
        
        std::string gateStr = std::to_string(gate.control) + "-" + std::to_string(gate.target);
        
        if (uniqueGates.find(gateStr) == uniqueGates.end()) {
            std::uniform_int_distribution<int> pos_dist(0, circuit.gates.size());
            int position = pos_dist(rng);
            
            circuit.gates.insert(circuit.gates.begin() + position, gate);
        }
    }
    
    if (uniform_dist(rng) < MUTATION_RATE && circuit.gates.size() > MIN_CIRCUIT_SIZE) {
        std::uniform_int_distribution<int> pos_dist(0, circuit.gates.size() - 1);
        int position = pos_dist(rng);
        
        std::string gateStr = std::to_string(circuit.gates[position].control) + "-" + 
                             std::to_string(circuit.gates[position].target);
        uniqueGates.erase(gateStr);
        
        circuit.gates.erase(circuit.gates.begin() + position);
    }
}

void printCircuit(const Circuit &circuit) {
    std::cout << "Circuit with " << circuit.gates.size() << " gates:\n";
    for (int i = 0; i < circuit.gates.size(); i++) {
        std::cout << "  Gate " << i << ": CX(control=" << circuit.gates[i].control 
                  << ", target=" << circuit.gates[i].target << ")\n";
    }
}