from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from collections import Counter

def add_key(circuit, plaintext_qubits, key_qubits):
    for i in range(8):
        circuit.cx(key_qubits[i], plaintext_qubits[i])
    return circuit

def sbox_4bit(circuit, qubits):
    circuit.cx(qubits[2], qubits[1])
    circuit.x(qubits[0])
    circuit.ccx(qubits[1], qubits[2], qubits[3])
    circuit.ccx(qubits[1], qubits[3], qubits[2])
    circuit.cx(qubits[3], qubits[1])
    circuit.x(qubits[3])
    circuit.ccx(qubits[2], qubits[0], qubits[1])
    circuit.cx(qubits[0], qubits[2])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[1], qubits[2])
    circuit.id(qubits[0])
    circuit.id(qubits[0])
    circuit.ccx(qubits[1], qubits[2], qubits[3])
    circuit.id(qubits[0])
    circuit.swap(qubits[2], qubits[3])
    return circuit

def sbox_layer(circuit, start_index, qubits):
    high_nibble = qubits[start_index:start_index+4]
    circuit = sbox_4bit(circuit, high_nibble)
    low_nibble = qubits[start_index+4:start_index+8] 
    circuit = sbox_4bit(circuit, low_nibble)
    return circuit

def permutation_layer(circuit, qubits):
    circuit.swap(qubits[7], qubits[0])
    circuit.swap(qubits[6], qubits[3])
    circuit.swap(qubits[6], qubits[5])
    circuit.swap(qubits[4], qubits[2])
    circuit.swap(qubits[4], qubits[1])
    return circuit

def inverse_permutation_layer(circuit, qubits):
    circuit.swap(qubits[4], qubits[1])
    circuit.swap(qubits[4], qubits[2])
    circuit.swap(qubits[6], qubits[5])
    circuit.swap(qubits[6], qubits[3])
    circuit.swap(qubits[7], qubits[0])
    return circuit

def key_permutation(circuit, qubits):
    circuit.swap(qubits[0], qubits[3])
    circuit.swap(qubits[2], qubits[3])
    circuit.swap(qubits[1], qubits[3])
    return circuit

def key_generation(circuit, key_qubits):
    high_nibble = key_qubits[0:4]
    low_nibble = key_qubits[4:8]
    circuit = sbox_4bit(circuit, high_nibble)
    circuit = sbox_4bit(circuit, low_nibble)
    circuit = key_permutation(circuit, high_nibble)
    circuit = key_permutation(circuit, low_nibble)
    return circuit

def inverse_sbox_4bit(circuit, qubits):
    circuit.swap(qubits[2], qubits[3])
    circuit.id(qubits[0])
    circuit.ccx(qubits[1], qubits[2], qubits[3])
    circuit.id(qubits[0])
    circuit.id(qubits[0])
    circuit.cx(qubits[1], qubits[2])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[0], qubits[2])
    circuit.ccx(qubits[2], qubits[0], qubits[1])
    circuit.x(qubits[3])
    circuit.cx(qubits[3], qubits[1])
    circuit.ccx(qubits[1], qubits[3], qubits[2])
    circuit.ccx(qubits[1], qubits[2], qubits[3])
    circuit.x(qubits[0])
    circuit.cx(qubits[2], qubits[1])
    return circuit

def inverse_sbox_layer(circuit, start_index, qubits):
    high_nibble = qubits[start_index:start_index+4]
    circuit = inverse_sbox_4bit(circuit, high_nibble)
    low_nibble = qubits[start_index+4:start_index+8]
    circuit = inverse_sbox_4bit(circuit, low_nibble)
    return circuit

def inverse_add_key(circuit, plaintext_qubits, key_qubits):
    for i in range(8):
        circuit.cx(key_qubits[i], plaintext_qubits[i])
    return circuit

def inverse_key_generation(circuit, key_qubits):
    low_nibble = key_qubits[4:8]
    high_nibble = key_qubits[0:4]
    circuit = inverse_key_permutation(circuit, low_nibble)
    circuit = inverse_key_permutation(circuit, high_nibble)
    circuit = inverse_sbox_4bit(circuit, low_nibble)
    circuit = inverse_sbox_4bit(circuit, high_nibble)
    return circuit

def inverse_key_permutation(circuit, qubits):
    circuit.swap(qubits[1], qubits[3])
    circuit.swap(qubits[2], qubits[3])
    circuit.swap(qubits[0], qubits[3])
    return circuit

def oracle_function(circuit, plaintext_qubits, key_qubits, target_qubit, ciphertext):
    circuit = add_key(circuit, plaintext_qubits, key_qubits)
    circuit.barrier()
    circuit = sbox_layer(circuit, 0, plaintext_qubits)
    circuit.barrier()
    circuit = permutation_layer(circuit, plaintext_qubits)
    circuit.barrier()
    circuit = key_generation(circuit, key_qubits)
    circuit.barrier()
    circuit = add_key(circuit, plaintext_qubits, key_qubits)
    circuit.barrier()
    circuit = sbox_layer(circuit, 0, plaintext_qubits)
    circuit.barrier()
    circuit = permutation_layer(circuit, plaintext_qubits)
    circuit.barrier()

    for i in range(8):
        if ciphertext[i] == '0':
            circuit.x(plaintext_qubits[i])

    gate = MCXGate(8)
    qubits_list = [plaintext_qubits[i] for i in range(8)]
    qubits_list.append(target_qubit)
    circuit.append(gate, qubits_list)

    for i in range(8):
        if ciphertext[i] == '0':
            circuit.x(plaintext_qubits[i])

    circuit.barrier()
    circuit = inverse_permutation_layer(circuit, plaintext_qubits)
    circuit.barrier()
    circuit = inverse_sbox_layer(circuit, 0, plaintext_qubits)
    circuit.barrier()
    circuit = inverse_add_key(circuit, plaintext_qubits, key_qubits)
    circuit.barrier()
    circuit = inverse_key_generation(circuit, key_qubits)
    circuit.barrier()
    circuit = inverse_permutation_layer(circuit, plaintext_qubits)
    circuit.barrier()
    circuit = inverse_sbox_layer(circuit, 0, plaintext_qubits)
    circuit.barrier()
    circuit = inverse_add_key(circuit, plaintext_qubits, key_qubits)
    circuit.barrier()

    return circuit

def diffusion(circuit, key_qubits):
    for qubit in key_qubits:
        circuit.h(qubit)

    for qubit in key_qubits:
        circuit.x(qubit)

    circuit.h(key_qubits[-1])
    
    gate = MCXGate(7)
    control_qubits = [key_qubits[i] for i in range(7)]
    target = key_qubits[7]
    circuit.append(gate, control_qubits + [target])
    
    circuit.h(key_qubits[-1])

    for qubit in key_qubits:
        circuit.x(qubit)

    for qubit in key_qubits:
        circuit.h(qubit)

    return circuit

def count_gates(circuit):
    gate_counts = Counter()
    for instruction, qargs, cargs in circuit.data:
        gate_name = instruction.name
        gate_counts[gate_name] += 1
    return gate_counts

def get_circuit_statistics(plaintext, ciphertext, iterations=1):
    key_register = QuantumRegister(8, 'key')
    plaintext_register = QuantumRegister(8, 'plaintext')
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(8, 'classical')
    
    circuit = QuantumCircuit(plaintext_register, key_register, target_register, classical_register)
    
    for i in range(8):
        if plaintext[i] == '1':
            circuit.x(plaintext_register[i])
    
    for qubit in key_register:
        circuit.h(qubit)
    
    for qubit in target_register:
        circuit.x(qubit)
        circuit.h(qubit)
    
    for _ in range(iterations):
        circuit = oracle_function(circuit, plaintext_register, key_register, target_register[0], ciphertext)
        circuit = diffusion(circuit, key_register)
    
    circuit.measure(key_register, classical_register)
    
    depth = circuit.depth()
    gate_counts = count_gates(circuit)
    total_gates = sum(gate_counts.values())
    
    return depth, gate_counts, total_gates

if __name__ == "__main__":
    plaintext = '10100110'
    ciphertext = '01111001'
    plaintext = plaintext[::-1]
    ciphertext = ciphertext[::-1]
    
    for iterations in [1, 3, 5, 12]:
        depth, gate_counts, total_gates = get_circuit_statistics(plaintext, ciphertext, iterations)
        
        print(f"\n===== CIRCUIT STATISTICS WITH {iterations} ITERATION(S) =====")
        print(f"Circuit depth: {depth}")
        print(f"Total gates: {total_gates}")
        print("\nGate counts:")
        
        sorted_gates = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)
        for gate_name, count in sorted_gates:
            print(f"  {gate_name}: {count}")