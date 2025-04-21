from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

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

def run_iterations(plaintext, ciphertext, max_iterations=12, shots=4096):
    output_dir = "quantum_iterations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_results = {}
    
    simulator = AerSimulator()
    
    for iteration in range(0, max_iterations + 1):
        print(f"Running iteration {iteration} of {max_iterations}...")
        
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
        
        for _ in range(iteration):
            circuit = oracle_function(circuit, plaintext_register, key_register, target_register[0], ciphertext)
            circuit = diffusion(circuit, key_register)
        
        circuit.measure(key_register, classical_register)
        
        job = simulator.run(circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        total_shots = sum(counts.values())
        percentages = {key: (value/total_shots)*100 for key, value in counts.items()}
        sorted_percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))
        
        all_results[iteration] = sorted_percentages
        
        plt.figure(figsize=(16, 10))
        ax = plot_histogram(sorted_percentages, title=f'Iteration {iteration}: Key Measurement Results', figsize=(16, 10))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"{output_dir}/histogram_iteration_{iteration}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        top_results = dict(list(sorted_percentages.items())[:5])
    
    plt.figure(figsize=(20, 15))
    
    for iteration in range(1, max_iterations + 1):
        plt.subplot(3, 4, iteration)
        top_results = dict(list(all_results[iteration].items())[:5])
        plot_histogram(top_results, title=f'Iteration {iteration}', figsize=(5, 4))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f"{output_dir}/combined_histograms.png", dpi=300, bbox_inches='tight')
    
    plt.figure(figsize=(18, 12))
    
    all_keys = set()
    for results in all_results.values():
        all_keys.update(results.keys())
    
    top_final_keys = list(all_results[max_iterations].keys())[:10]
    
    for key in top_final_keys:
        probabilities = []
        for i in range(1, max_iterations + 1):
            if key in all_results[i]:
                probabilities.append(all_results[i][key])
            else:
                probabilities.append(0)
        plt.plot(range(1, max_iterations + 1), probabilities, marker='o', linewidth=3, markersize=10, label=key)
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Probability (%)', fontsize=16)
    plt.title('Evolution of Key Probabilities Across Iterations', fontsize=20)
    plt.xticks(range(1, max_iterations + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.savefig(f"{output_dir}/probability_evolution.png", dpi=300, bbox_inches='tight')
    
    # Find keys that exceed 15% probability in any iteration
    high_probability_keys = set()
    for iteration, results in all_results.items():
        for key, probability in results.items():
            if probability > 15:
                high_probability_keys.add(key)
    
    # Generate cumulative probability graph for these high-probability keys
    plt.figure(figsize=(18, 12))
    
    for key in high_probability_keys:
        cumulative_probabilities = []
        running_sum = 0
        for i in range(1, max_iterations + 1):
            probability = all_results[i].get(key, 0)
            running_sum += probability
            cumulative_probabilities.append(running_sum)
        
        plt.plot(range(1, max_iterations + 1), cumulative_probabilities, marker='o', 
                 linewidth=3, markersize=10, label=f"Key {key} (Cumulative)")
    
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Cumulative Probability (%)', fontsize=16)
    plt.title('Cumulative Probability Growth of High-Probability Keys', fontsize=20)
    plt.xticks(range(1, max_iterations + 1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.savefig(f"{output_dir}/cumulative_probability_growth.png", dpi=300, bbox_inches='tight')
    
    print(f"\nAll histograms have been saved to the '{output_dir}' directory")
    print(f"Combined visualization saved as '{output_dir}/combined_histograms.png'")
    print(f"Probability evolution graph saved as '{output_dir}/probability_evolution.png'")
    print(f"Cumulative probability growth graph saved as '{output_dir}/cumulative_probability_growth.png'")
    
    return all_results

if __name__ == "__main__":
    plaintext = '11000101'
    ciphertext = '00100110'
    plaintext = plaintext[::-1]
    ciphertext = ciphertext[::-1]
    max_iterations = 12
    shots = 1000000
    
    results = run_iterations(plaintext, ciphertext, max_iterations, shots)
    
    print("\nTop 3 most likely keys after", max_iterations, "iterations:")
    final_results = results[max_iterations]
    for i, (key, probability) in enumerate(list(final_results.items())[:6]):
        print(f"{i+1}. {key}: {probability:.2f}%")