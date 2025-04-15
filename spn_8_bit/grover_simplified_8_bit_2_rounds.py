from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

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

def generate_circuit_diagram(plaintext, ciphertext, num_iterations=1):
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

    circuit = oracle_function(circuit, plaintext_register, key_register, target_register[0], ciphertext)
    circuit = diffusion(circuit, key_register)

    circuit.measure(key_register, classical_register)
    
    figure = circuit.draw(output='mpl', fold=100, scale=0.7)
    return figure

def main(plaintext, ciphertext, num_iterations):
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

    for _ in range(num_iterations):
        circuit = oracle_function(circuit, plaintext_register, key_register, target_register[0], ciphertext)
        circuit = diffusion(circuit, key_register)

    circuit.measure(key_register, classical_register)
    
    circuit_fig = circuit.draw(output='mpl', fold=100, scale=0.7)
    circuit_fig.savefig('yo_yo_circuit.png', dpi=300, bbox_inches='tight')

    simulator = AerSimulator()
    job = simulator.run(circuit, shots=10000)
    result = job.result()
    counts = result.get_counts()

    total_shots = sum(counts.values())
    percentages = {key: (value/total_shots)*100 for key, value in counts.items()}
    sorted_percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

    histogram_fig = plot_histogram(sorted_percentages, figsize=(12, 6), title='Key Measurement Results')
    histogram_fig.savefig('yo_yo_histogram.png', dpi=300, bbox_inches='tight')

    return sorted_percentages, circuit_fig, histogram_fig

plaintext = '11011011'
ciphertext = '01000100'

fig = generate_circuit_diagram(plaintext, ciphertext)
fig.savefig('yo_yo_circuit_visualization.png', dpi=300, bbox_inches='tight')

num_iterations = 12
results, circuit_fig, histogram_fig = main(plaintext, ciphertext, num_iterations)

print("\nMeasurement results (percentage):")
for state, percentage in results.items():
    print(f"{state}: {percentage:.2f}%")

print("\nCircuit visualization and histogram have been saved as 'yo_yo_circuit_visualization.png', 'yo_yo_circuit.png', and 'yo_yo_histogram.png'")