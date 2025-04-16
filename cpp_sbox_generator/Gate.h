#ifndef GATE_H
#define GATE_H

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

enum class GateType {
    NOT,
    CNOT,
    TOFFOLI,
    FREDKIN,
    SWAP,
    MULTI_CONTROLLED_X
};

struct Gate {
    std::string name;
    GateType type;
    std::vector<int> qubits;
    
    Gate(GateType t, const std::vector<int>& q, const std::string& gate_name = "") 
        : name(gate_name), type(t), qubits(q) {}
    
    // Optional metadata for more complex gate representations
    std::unordered_map<std::string, double> parameters;
    
    // Method to add additional parameters
    void add_parameter(const std::string& key, double value) {
        parameters[key] = value;
    }
    
    // Method to get a parameter with a default value
    double get_parameter(const std::string& key, double default_value = 0.0) const {
        auto it = parameters.find(key);
        return (it != parameters.end()) ? it->second : default_value;
    }
};

struct GateDefinition {
    std::string name;
    int qubits;
    int controls;
    int quantum_cost;
    int two_qubit_cost;
    int gate_count;
    
    // Lambda function to apply the gate to a quantum circuit
    std::function<void(class QuantumCircuit&, const std::vector<int>&)> apply;
    
    // Optional additional metadata
    std::unordered_map<std::string, std::string> metadata;
    
    // Method to add metadata
    void add_metadata(const std::string& key, const std::string& value) {
        metadata[key] = value;
    }
    
    // Method to get metadata with a default value
    std::string get_metadata(const std::string& key, const std::string& default_value = "") const {
        auto it = metadata.find(key);
        return (it != metadata.end()) ? it->second : default_value;
    }
    
    // Comparison operator for potential sorting or unique identification
    bool operator==(const GateDefinition& other) const {
        return name == other.name && 
               qubits == other.qubits && 
               controls == other.controls;
    }
};

// Optional hash function for GateDefinition to use in unordered containers
namespace std {
    template <>
    struct hash<GateDefinition> {
        size_t operator()(const GateDefinition& gate) const {
            return hash<string>()(gate.name) ^ 
                   hash<int>()(gate.qubits) ^ 
                   hash<int>()(gate.controls);
        }
    };
}

#endif // GATE_H