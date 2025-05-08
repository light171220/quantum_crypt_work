#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <string>
#include <set>
#include <thread>
#include <atomic>
#include <utility>
#include <iomanip>

enum class GateType
{
    X,
    CX,
    CCX,
    SWAP
};

struct Gate
{
    GateType type;
    std::vector<int> qubits;

    Gate() : type(GateType::X) { qubits.push_back(0); }
    Gate(GateType t, const std::vector<int> &q) : type(t), qubits(q) {}

    std::string toString() const
    {
        std::string result;
        switch (type)
        {
        case GateType::X:
            result = "X(" + std::to_string(qubits[0]) + ")";
            break;
        case GateType::CX:
            result = "CX(" + std::to_string(qubits[0]) + "," + std::to_string(qubits[1]) + ")";
            break;
        case GateType::CCX:
            result = "CCX(" + std::to_string(qubits[0]) + "," + std::to_string(qubits[1]) + "," + std::to_string(qubits[2]) + ")";
            break;
        case GateType::SWAP:
            result = "SWAP(" + std::to_string(qubits[0]) + "," + std::to_string(qubits[1]) + ")";
            break;
        }
        return result;
    }
};

using Circuit = std::vector<Gate>;

struct Individual
{
    Circuit circuit;
    double fitness;
    int correctCount;
    bool isReversible;

    Individual() : fitness(0.0), correctCount(0), isReversible(false) {}
    Individual(const Circuit &c) : circuit(c), fitness(0.0), correctCount(0), isReversible(false) {}
};

class GeneticAlgorithm
{
private:
    int numQubits;
    int maxGates;
    int populationSize;
    double mutationRate;
    double crossoverRate;
    int tournamentSize;
    int maxGenerations;
    bool verbose;

    // Updated s0Box according to the provided values
    std::array<std::array<int, 4>, 4> s0Box = {{
        {{1, 0, 3, 2}},
        {{3, 2, 1, 0}},
        {{0, 2, 1, 3}},
        {{3, 1, 3, 2}}
    }};

    std::mt19937 rng;

public:
    GeneticAlgorithm(int nQubits, int mGates, int popSize, double mutRate,
                     double crossRate, int tournSize, int maxGen, bool verb = true)
        : numQubits(nQubits), maxGates(mGates), populationSize(popSize),
          mutationRate(mutRate), crossoverRate(crossRate),
          tournamentSize(tournSize), maxGenerations(maxGen), verbose(verb)
    {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        rng = std::mt19937(seed);
    }

    void applyGate(std::bitset<4> &state, const Gate &gate)
    {
        switch (gate.type)
        {
        case GateType::X:
            state.flip(gate.qubits[0]);
            break;
        case GateType::CX:
            if (state[gate.qubits[0]])
                state.flip(gate.qubits[1]);
            break;
        case GateType::CCX:
            if (state[gate.qubits[0]] && state[gate.qubits[1]])
                state.flip(gate.qubits[2]);
            break;
        case GateType::SWAP:
        {
            bool temp = state[gate.qubits[0]];
            state[gate.qubits[0]] = state[gate.qubits[1]];
            state[gate.qubits[1]] = temp;
            break;
        }
        }
    }

    std::bitset<4> applyCircuit(const Circuit &circuit, std::bitset<4> input)
    {
        std::bitset<4> state = input;
        for (const Gate &gate : circuit)
            applyGate(state, gate);
        return state;
    }

    bool checkReversibility(const Circuit &circuit)
    {
        std::unordered_map<std::string, int> outputCounts;
        
        for (int input = 0; input < 16; input++)
        {
            std::bitset<4> inputBits(input);
            std::bitset<4> outputState = applyCircuit(circuit, inputBits);
            std::string outputStr = outputState.to_string();
            
            if (outputCounts.find(outputStr) != outputCounts.end())
            {
                return false;
            }
            outputCounts[outputStr] = 1;
        }
        
        return true;
    }

    std::pair<double, int> evaluateCircuit(const Circuit &circuit)
    {
        int correctOutputs = 0;
        int totalCases = 16;
        bool isReversible = checkReversibility(circuit);

        for (int input = 0; input < 16; input++)
        {
            std::bitset<4> inputBits(input);
            int row = (inputBits[0] << 1) | inputBits[3];
            int col = (inputBits[1] << 1) | inputBits[2];
            int expectedOutput = s0Box[row][col];
            std::bitset<2> expectedBits(expectedOutput);
            std::bitset<4> outputState = applyCircuit(circuit, inputBits);
            std::bitset<2> actualBits;
            actualBits[0] = outputState[0];
            actualBits[1] = outputState[1];
            if (actualBits == expectedBits)
                correctOutputs++;
        }

        double fitness = (double)correctOutputs / totalCases;
        if (correctOutputs == 16)
        {
            fitness += 1.0 - (double)circuit.size() / (2.0 * maxGates);
            if (isReversible)
            {
                fitness += 0.5;
            }
        }
        return std::make_pair(fitness, correctOutputs);
    }

    Gate randomGate()
    {
        std::uniform_int_distribution<int> typeDist(0, 3);
        int type = typeDist(rng);
        std::vector<int> qubits;
        std::uniform_int_distribution<int> qubitDist(0, numQubits - 1);

        switch (static_cast<GateType>(type))
        {
        case GateType::X:
            qubits.push_back(qubitDist(rng));
            break;
        case GateType::CX:
        {
            int control = qubitDist(rng);
            int target = qubitDist(rng);
            while (target == control)
                target = qubitDist(rng);
            qubits.push_back(control);
            qubits.push_back(target);
            break;
        }
        case GateType::CCX:
        {
            std::set<int> qubitSet;
            while (qubitSet.size() < 3)
                qubitSet.insert(qubitDist(rng));
            qubits.assign(qubitSet.begin(), qubitSet.end());
            break;
        }
        case GateType::SWAP:
        {
            int q1 = qubitDist(rng);
            int q2 = qubitDist(rng);
            while (q2 == q1)
                q2 = qubitDist(rng);
            qubits.push_back(q1);
            qubits.push_back(q2);
            break;
        }
        }
        return Gate(static_cast<GateType>(type), qubits);
    }

    Circuit randomCircuit()
    {
        std::uniform_int_distribution<int> sizeDist(1, maxGates);
        int size = sizeDist(rng);
        Circuit circuit;
        for (int i = 0; i < size; i++)
            circuit.push_back(randomGate());
        return circuit;
    }

    Circuit seedCircuit()
    {
        // This is a seed circuit that might help for the new s0Box
        Circuit circuit;
        circuit.push_back(Gate(GateType::X, {0}));
        circuit.push_back(Gate(GateType::CCX, {0, 1, 2}));
        circuit.push_back(Gate(GateType::CCX, {2, 3, 1}));
        circuit.push_back(Gate(GateType::CX, {0, 3}));
        circuit.push_back(Gate(GateType::CCX, {0, 2, 1}));
        circuit.push_back(Gate(GateType::SWAP, {1, 0}));
        circuit.push_back(Gate(GateType::CX, {3, 2}));
        circuit.push_back(Gate(GateType::CCX, {1, 3, 0}));
        return circuit;
    }

    std::vector<Individual> initializePopulation()
    {
        std::vector<Individual> population;
        Individual seeded(seedCircuit());
        auto evalResult = evaluateCircuit(seeded.circuit);
        seeded.fitness = evalResult.first;
        seeded.correctCount = evalResult.second;
        seeded.isReversible = checkReversibility(seeded.circuit);
        population.push_back(seeded);

        for (int i = 1; i < populationSize; i++)
        {
            Individual ind(randomCircuit());
            auto evalResult = evaluateCircuit(ind.circuit);
            ind.fitness = evalResult.first;
            ind.correctCount = evalResult.second;
            ind.isReversible = checkReversibility(ind.circuit);
            population.push_back(ind);
        }
        return population;
    }

    Individual selectParent(const std::vector<Individual> &population)
    {
        std::uniform_int_distribution<int> dist(0, population.size() - 1);
        Individual best;
        best.fitness = -1.0;
        for (int i = 0; i < tournamentSize; i++)
        {
            int idx = dist(rng);
            if (population[idx].fitness > best.fitness)
                best = population[idx];
        }
        return best;
    }

    Individual crossover(const Individual &parent1, const Individual &parent2)
    {
        std::uniform_real_distribution<double> realDist(0.0, 1.0);
        if (realDist(rng) > crossoverRate)
            return parent1;

        std::uniform_int_distribution<int> dist1(0, parent1.circuit.size());
        std::uniform_int_distribution<int> dist2(0, parent2.circuit.size());
        int cutPoint1 = dist1(rng);
        int cutPoint2 = dist2(rng);

        Circuit childCircuit;
        for (int i = 0; i < cutPoint1; i++)
            childCircuit.push_back(parent1.circuit[i]);
        for (int i = cutPoint2; i < parent2.circuit.size(); i++)
            childCircuit.push_back(parent2.circuit[i]);

        if (childCircuit.size() > maxGates)
            childCircuit.erase(childCircuit.begin() + maxGates, childCircuit.end());

        Individual child(childCircuit);
        auto evalResult = evaluateCircuit(child.circuit);
        child.fitness = evalResult.first;
        child.correctCount = evalResult.second;
        child.isReversible = checkReversibility(child.circuit);
        return child;
    }

    void mutate(Individual &ind)
    {
        std::uniform_real_distribution<double> realDist(0.0, 1.0);
        double adjustedMutRate = mutationRate * (1.0 + (ind.fitness < 0.9 ? 0.5 : 0.0));

        for (int i = 0; i < ind.circuit.size(); i++)
        {
            if (realDist(rng) < adjustedMutRate)
                ind.circuit[i] = randomGate();
        }

        if (realDist(rng) < adjustedMutRate && ind.circuit.size() < maxGates)
        {
            ind.circuit.push_back(randomGate());
        }

        if (realDist(rng) < adjustedMutRate && !ind.circuit.empty())
        {
            std::uniform_int_distribution<int> dist(0, ind.circuit.size() - 1);
            int idx = dist(rng);
            ind.circuit.erase(ind.circuit.begin() + idx);
        }

        if (realDist(rng) < adjustedMutRate && ind.circuit.size() >= 2)
        {
            std::uniform_int_distribution<int> dist(0, ind.circuit.size() - 1);
            int idx1 = dist(rng);
            int idx2 = dist(rng);
            while (idx2 == idx1)
                idx2 = dist(rng);
            std::swap(ind.circuit[idx1], ind.circuit[idx2]);
        }

        auto evalResult = evaluateCircuit(ind.circuit);
        ind.fitness = evalResult.first;
        ind.correctCount = evalResult.second;
        ind.isReversible = checkReversibility(ind.circuit);
    }

    void targetedMutate(Individual &ind)
    {
        Circuit backup = ind.circuit;
        double backupFitness = ind.fitness;
        int backupCorrectCount = ind.correctCount;
        bool backupReversible = ind.isReversible;

        for (int attempts = 0; attempts < 10; attempts++)
        {
            int idx = std::uniform_int_distribution<int>(0, ind.circuit.size() - 1)(rng);
            Gate originalGate = ind.circuit[idx];

            for (int i = 0; i < 5; i++)
            {
                ind.circuit[idx] = randomGate();
                auto evalResult = evaluateCircuit(ind.circuit);
                ind.isReversible = checkReversibility(ind.circuit);
                if (evalResult.second > backupCorrectCount ||
                    (evalResult.second == backupCorrectCount && evalResult.first > backupFitness))
                {
                    ind.fitness = evalResult.first;
                    ind.correctCount = evalResult.second;
                    return;
                }
            }

            ind.circuit[idx] = originalGate;
        }

        ind.fitness = backupFitness;
        ind.correctCount = backupCorrectCount;
        ind.isReversible = backupReversible;
    }

    Circuit run()
    {
        std::vector<Individual> population = initializePopulation();
        std::sort(population.begin(), population.end(),
                  [](const Individual &a, const Individual &b)
                  { return a.fitness > b.fitness; });

        Individual bestEver = population[0];
        if (verbose)
        {
            std::cout << "Initial best fitness: " << bestEver.fitness << " ("
                      << bestEver.correctCount << "/16 correct), Reversible: "
                      << (bestEver.isReversible ? "Yes" : "No") << std::endl;
        }

        int stagnantGenerations = 0;

        for (int generation = 0; generation < maxGenerations; generation++)
        {
            std::vector<Individual> newPopulation;

            int eliteCount = std::max(1, populationSize / 10);
            for (int i = 0; i < eliteCount; i++)
                newPopulation.push_back(population[i]);

            for (int i = 0; i < std::min(5, (int)population.size()); i++)
            {
                if (population[i].correctCount >= 14)
                {
                    Individual improved = population[i];
                    targetedMutate(improved);
                    if (improved.correctCount > population[i].correctCount)
                    {
                        newPopulation.push_back(improved);
                    }
                }
            }

            while (newPopulation.size() < populationSize)
            {
                Individual parent1 = selectParent(population);
                Individual parent2 = selectParent(population);
                Individual child = crossover(parent1, parent2);
                mutate(child);
                newPopulation.push_back(child);
            }

            population = newPopulation;
            std::sort(population.begin(), population.end(),
                      [](const Individual &a, const Individual &b)
                      { return a.fitness > b.fitness; });

            bool improved = false;
            if (population[0].correctCount > bestEver.correctCount ||
                (population[0].correctCount == bestEver.correctCount &&
                 population[0].fitness > bestEver.fitness))
            {
                bestEver = population[0];
                improved = true;
                stagnantGenerations = 0;
            }
            else
            {
                stagnantGenerations++;
            }

            if (verbose && (improved || generation % 100 == 0))
            {
                std::cout << "Generation " << generation << ", Best fitness: "
                          << bestEver.fitness << ", Circuit size: "
                          << bestEver.circuit.size() << ", Correct: "
                          << bestEver.correctCount << "/16, Reversible: "
                          << (bestEver.isReversible ? "Yes" : "No") << std::endl;
            }

            if (bestEver.correctCount == 16 && bestEver.isReversible)
            {
                if (verbose)
                    std::cout << "Found perfect reversible solution at generation " << generation << std::endl;
                break;
            }

            if (stagnantGenerations > 200)
            {
                if (verbose)
                    std::cout << "Evolution stagnant, refreshing population at generation " << generation << std::endl;
                std::vector<Individual> newPop;
                newPop.push_back(bestEver);
                for (int i = 1; i < populationSize; i++)
                {
                    Individual ind(randomCircuit());
                    auto evalResult = evaluateCircuit(ind.circuit);
                    ind.fitness = evalResult.first;
                    ind.correctCount = evalResult.second;
                    ind.isReversible = checkReversibility(ind.circuit);
                    newPop.push_back(ind);
                }
                population = newPop;
                std::sort(population.begin(), population.end(),
                          [](const Individual &a, const Individual &b)
                          { return a.fitness > b.fitness; });
                stagnantGenerations = 0;
            }
        }

        if (verbose)
        {
            std::cout << "Evolution complete. Best fitness: " << bestEver.fitness
                      << ", Correct: " << bestEver.correctCount << "/16"
                      << ", Reversible: " << (bestEver.isReversible ? "Yes" : "No")
                      << std::endl;
        }

        optimizeCircuit(bestEver.circuit);
        return bestEver.circuit;
    }

    void optimizeCircuit(Circuit &circuit)
    {
        if (circuit.empty())
            return;
        auto originalEval = evaluateCircuit(circuit);
        double originalFitness = originalEval.first;
        int originalCorrect = originalEval.second;
        bool originalReversible = checkReversibility(circuit);

        for (int i = 0; i < circuit.size(); i++)
        {
            Gate removedGate = circuit[i];
            circuit.erase(circuit.begin() + i);
            auto newEval = evaluateCircuit(circuit);
            bool newReversible = checkReversibility(circuit);
            if (newEval.second < originalCorrect || 
                (newEval.second == originalCorrect && !newReversible && originalReversible))
            {
                circuit.insert(circuit.begin() + i, removedGate);
            }
            else
            {
                originalCorrect = newEval.second;
                originalReversible = newReversible;
                i--;
            }
        }
    }

    void verifyCircuit(const Circuit &circuit)
    {
        std::cout << "\nVerifying circuit...\n";
        std::cout << "Input  | Expected | Actual | Row,Col | Match\n";
        std::cout << "-------|----------|--------|---------|------\n";

        int matchCount = 0;
        bool isReversible = true;
        std::unordered_map<std::string, int> outputCounts;

        for (int input = 0; input < 16; input++)
        {
            std::bitset<4> inputBits(input);
            int row = (inputBits[0] << 1) | inputBits[3];
            int col = (inputBits[1] << 1) | inputBits[2];
            int expectedOutput = s0Box[row][col];
            std::bitset<2> expectedBits(expectedOutput);
            std::bitset<4> outputState = applyCircuit(circuit, inputBits);
            std::bitset<2> actualBits;
            actualBits[0] = outputState[0];
            actualBits[1] = outputState[1];
            bool match = (actualBits == expectedBits);
            if (match)
                matchCount++;

            std::cout << inputBits << " | " << expectedBits << "      | "
                      << actualBits << "    | " << row << "," << col << "    | "
                      << (match ? "Yes" : "No") << "\n";

            // Check for reversibility
            std::string outputStr = outputState.to_string();
            if (outputCounts.find(outputStr) != outputCounts.end())
            {
                isReversible = false;
            }
            outputCounts[outputStr] = 1;
        }

        std::cout << "\nTotal matches: " << matchCount << "/16 ("
                  << (matchCount / 16.0 * 100) << "%)\n";
        
        std::cout << "Reversibility: " << (isReversible ? "Yes (all outputs unique)" : "No (duplicate outputs found)") << "\n";

        if (matchCount < 16)
        {
            std::cout << "\nIncorrect mappings:\n";
            for (int input = 0; input < 16; input++)
            {
                std::bitset<4> inputBits(input);
                int row = (inputBits[0] << 1) | inputBits[3];
                int col = (inputBits[1] << 1) | inputBits[2];
                int expectedOutput = s0Box[row][col];
                std::bitset<2> expectedBits(expectedOutput);
                std::bitset<4> outputState = applyCircuit(circuit, inputBits);
                std::bitset<2> actualBits;
                actualBits[0] = outputState[0];
                actualBits[1] = outputState[1];
                bool match = (actualBits == expectedBits);
                if (!match)
                {
                    std::cout << "Input: " << inputBits << " (Row " << row << ", Col " << col
                              << "), Expected: " << expectedBits << ", Got: " << actualBits << std::endl;
                }
            }
        }
    }

    std::string generateQiskitCode(const Circuit &circuit)
    {
        std::string code = "from qiskit import QuantumCircuit\n\n";
        code += "def s0_box_circuit():\n";
        code += "    qc = QuantumCircuit(4)\n\n";

        for (const Gate &gate : circuit)
        {
            switch (gate.type)
            {
            case GateType::X:
                code += "    qc.x(" + std::to_string(gate.qubits[0]) + ")\n";
                break;
            case GateType::CX:
                code += "    qc.cx(" + std::to_string(gate.qubits[0]) + ", " + std::to_string(gate.qubits[1]) + ")\n";
                break;
            case GateType::CCX:
                code += "    qc.ccx(" + std::to_string(gate.qubits[0]) + ", " + std::to_string(gate.qubits[1]) + ", " + std::to_string(gate.qubits[2]) + ")\n";
                break;
            case GateType::SWAP:
                code += "    qc.swap(" + std::to_string(gate.qubits[0]) + ", " + std::to_string(gate.qubits[1]) + ")\n";
                break;
            }
        }

        code += "\n    return qc\n\n";
        code += "qc = s0_box_circuit()\n";
        code += "print(qc.draw())\n";
        return code;
    }

    void saveCircuit(const Circuit &circuit, const std::string &filename)
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            for (const Gate &gate : circuit)
                file << gate.toString() << std::endl;
            file.close();
        }
    }

    void saveQiskitCode(const Circuit &circuit, const std::string &filename)
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << generateQiskitCode(circuit);
            file.close();
        }
    }
};

int main()
{
    // We can increase the population size and max generations for better results
    GeneticAlgorithm ga(4, 30, 300, 0.3, 0.9, 5, 5000);
    Circuit bestCircuit = ga.run();
    ga.verifyCircuit(bestCircuit);

    std::cout << "\nBest circuit found:\n";
    for (const Gate &gate : bestCircuit)
        std::cout << gate.toString() << std::endl;

    ga.saveCircuit(bestCircuit, "s0_box_circuit.txt");
    ga.saveQiskitCode(bestCircuit, "s0_box_qiskit.py");

    return 0;
}