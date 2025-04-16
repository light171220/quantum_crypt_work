#ifndef QUANTUM_SIMULATOR_H
#define QUANTUM_SIMULATOR_H

#include <random>
#include "QuantumCircuit.h"

class QuantumSimulator {
private:
    std::mt19937 rng;
    bool collect_stats;
    
public:
    struct GateMetrics {
        int total_gates = 0;
        int not_gates = 0;
        int cnot_gates = 0;
        int toffoli_gates = 0;
        int swap_gates = 0;
        int fredkin_gates = 0;
        int multi_controlled_gates = 0;
        int quantum_cost = 0;
    };
    
    GateMetrics metrics;

    QuantumSimulator(unsigned int seed = 42, bool collect_stats = false);
    
    void reset_metrics();
    uint64_t simulate(const QuantumCircuit& circuit, uint64_t input_state);
};

#endif // QUANTUM_SIMULATOR_H