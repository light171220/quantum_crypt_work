#include "QuantumCircuit.h"

QuantumCircuit::QuantumCircuit(int qubits, int cbits) : num_qubits(qubits), num_classical_bits(cbits) {}

QuantumCircuit QuantumCircuit::copy() const {
    return *this;
}

void QuantumCircuit::x(int qubit) {
    gates.emplace_back(GateType::NOT, std::vector<int>{qubit});
}

void QuantumCircuit::cx(int control, int target) {
    gates.emplace_back(GateType::CNOT, std::vector<int>{control, target});
}

void QuantumCircuit::ccx(int control1, int control2, int target) {
    gates.emplace_back(GateType::TOFFOLI, std::vector<int>{control1, control2, target});
}

void QuantumCircuit::swap(int qubit1, int qubit2) {
    gates.emplace_back(GateType::SWAP, std::vector<int>{qubit1, qubit2});
}

void QuantumCircuit::fredkin(int control, int target1, int target2) {
    gates.emplace_back(GateType::FREDKIN, std::vector<int>{control, target1, target2});
}

void QuantumCircuit::multi_controlled_x(const std::vector<int>& controls, int target) {
    std::vector<int> qubits = controls;
    qubits.push_back(target);
    gates.emplace_back(GateType::MULTI_CONTROLLED_X, qubits);
}

int QuantumCircuit::depth() const {
    return gates.size();
}

int QuantumCircuit::size() const {
    return gates.size();
}

void QuantumCircuit::clear() {
    gates.clear();
}

void QuantumCircuit::append(const QuantumCircuit& other) {
    for (const auto& gate : other.gates) {
        gates.push_back(gate);
    }
}

void QuantumCircuit::optimize() {
    for (size_t i = 0; i < gates.size(); i++) {
        if (gates[i].type == GateType::NOT && i + 1 < gates.size() && 
            gates[i+1].type == GateType::NOT && 
            gates[i].qubits[0] == gates[i+1].qubits[0]) {
            gates.erase(gates.begin() + i, gates.begin() + i + 2);
            i--;
        }
    }
    
    for (size_t i = 0; i < gates.size(); i++) {
        if (gates[i].type == GateType::SWAP && i + 1 < gates.size() && 
            gates[i+1].type == GateType::SWAP && 
            gates[i].qubits[0] == gates[i+1].qubits[0] &&
            gates[i].qubits[1] == gates[i+1].qubits[1]) {
            gates.erase(gates.begin() + i, gates.begin() + i + 2);
            i--;
        }
    }
}