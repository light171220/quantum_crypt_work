#include "QuantumSimulator.h"

QuantumSimulator::QuantumSimulator(unsigned int seed, bool collect_stats)
    : rng(seed), collect_stats(collect_stats) {}

void QuantumSimulator::reset_metrics() {
    metrics = GateMetrics();
}

uint64_t QuantumSimulator::simulate(const QuantumCircuit& circuit, uint64_t input_state) {
    if (collect_stats) {
        reset_metrics();
    }
    
    uint64_t state = input_state;
    
    for (const auto& gate : circuit.gates) {
        if (collect_stats) {
            metrics.total_gates++;
            
            switch(gate.type) {
                case GateType::NOT:
                    metrics.not_gates++;
                    metrics.quantum_cost += 1;
                    break;
                case GateType::CNOT:
                    metrics.cnot_gates++;
                    metrics.quantum_cost += 1;
                    break;
                case GateType::TOFFOLI:
                    metrics.toffoli_gates++;
                    metrics.quantum_cost += 5;
                    break;
                case GateType::SWAP:
                    metrics.swap_gates++;
                    metrics.quantum_cost += 3;
                    break;
                case GateType::FREDKIN:
                    metrics.fredkin_gates++;
                    metrics.quantum_cost += 7;
                    break;
                case GateType::MULTI_CONTROLLED_X:
                    metrics.multi_controlled_gates++;
                    metrics.quantum_cost += (gate.qubits.size() - 1) * 4;
                    break;
            }
        }
        
        switch (gate.type) {
            case GateType::NOT: {
                int qubit = gate.qubits[0];
                state ^= (1ULL << qubit);
                break;
            }
            case GateType::CNOT: {
                int control = gate.qubits[0];
                int target = gate.qubits[1];
                
                if ((state & (1ULL << control)) != 0) {
                    state ^= (1ULL << target);
                }
                break;
            }
            case GateType::TOFFOLI: {
                int control1 = gate.qubits[0];
                int control2 = gate.qubits[1];
                int target = gate.qubits[2];
                
                if (((state & (1ULL << control1)) != 0) && ((state & (1ULL << control2)) != 0)) {
                    state ^= (1ULL << target);
                }
                break;
            }
            case GateType::SWAP: {
                int qubit1 = gate.qubits[0];
                int qubit2 = gate.qubits[1];
                
                bool val1 = (state & (1ULL << qubit1)) != 0;
                bool val2 = (state & (1ULL << qubit2)) != 0;
                
                if (val1 != val2) {
                    state ^= (1ULL << qubit1);
                    state ^= (1ULL << qubit2);
                }
                break;
            }
            case GateType::FREDKIN: {
                int control = gate.qubits[0];
                int target1 = gate.qubits[1];
                int target2 = gate.qubits[2];
                
                if ((state & (1ULL << control)) != 0) {
                    bool val1 = (state & (1ULL << target1)) != 0;
                    bool val2 = (state & (1ULL << target2)) != 0;
                    
                    if (val1 != val2) {
                        state ^= (1ULL << target1);
                        state ^= (1ULL << target2);
                    }
                }
                break;
            }
            case GateType::MULTI_CONTROLLED_X: {
                int target = gate.qubits.back();
                bool all_controls_set = true;
                
                for (size_t i = 0; i < gate.qubits.size() - 1; i++) {
                    if ((state & (1ULL << gate.qubits[i])) == 0) {
                        all_controls_set = false;
                        break;
                    }
                }
                
                if (all_controls_set) {
                    state ^= (1ULL << target);
                }
                break;
            }
        }
    }
    
    return state;
}