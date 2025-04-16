#ifndef QUANTUM_CIRCUIT_H
#define QUANTUM_CIRCUIT_H

#include <vector>
#include "Gate.h"

class QuantumCircuit {
public:
    int num_qubits;
    int num_classical_bits;
    std::vector<Gate> gates;
    
    // Default constructor
    QuantumCircuit() : num_qubits(0), num_classical_bits(0) {}
    
    // Constructor with parameters
    QuantumCircuit(int qubits, int cbits);
    
    QuantumCircuit copy() const;
    
    void x(int qubit);
    void cx(int control, int target);
    void ccx(int control1, int control2, int target);
    void swap(int qubit1, int qubit2);
    void fredkin(int control, int target1, int target2);
    void multi_controlled_x(const std::vector<int>& controls, int target);
    
    int depth() const;
    int size() const;
    void clear();
    void append(const QuantumCircuit& other);
    void optimize();
};

#endif // QUANTUM_CIRCUIT_H