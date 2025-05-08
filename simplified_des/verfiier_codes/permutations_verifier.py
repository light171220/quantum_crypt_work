from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def build_initial_permutation(circuit, qubits):
    circuit.swap(qubits[4], qubits[3])
    circuit.swap(qubits[3], qubits[7])
    circuit.swap(qubits[0], qubits[1])
    circuit.swap(qubits[3], qubits[1])
    circuit.swap(qubits[5], qubits[1])
    circuit.swap(qubits[6], qubits[7])

def build_inverse_initial_permutation(circuit, qubits):
    circuit.swap(qubits[6], qubits[7])
    circuit.swap(qubits[5], qubits[1])
    circuit.swap(qubits[3], qubits[1])
    circuit.swap(qubits[0], qubits[1])
    circuit.swap(qubits[3], qubits[7])
    circuit.swap(qubits[4], qubits[3])

def build_final_permutation(circuit, qubits):
    circuit.swap(qubits[0], qubits[6])
    circuit.swap(qubits[6], qubits[1])
    circuit.swap(qubits[0], qubits[3])
    circuit.swap(qubits[6], qubits[5])
    circuit.swap(qubits[6], qubits[7])
    circuit.swap(qubits[4], qubits[3])

def build_inverse_final_permutation(circuit, qubits):
    circuit.swap(qubits[4], qubits[3])
    circuit.swap(qubits[6], qubits[7])
    circuit.swap(qubits[6], qubits[5])
    circuit.swap(qubits[0], qubits[3])
    circuit.swap(qubits[6], qubits[1])
    circuit.swap(qubits[0], qubits[6])

def classical_initial_permutation(bits):
    ip = [2, 6, 3, 1, 4, 8, 5, 7]
    return tuple(bits[i-1] for i in ip)

def classical_final_permutation(bits):
    fp = [4, 1, 3, 5, 7, 2, 8, 6]
    return tuple(bits[i-1] for i in fp)

def verify_permutation(perm_name, build_perm_func, classical_perm_func=None):
    print(f"Verifying {perm_name}...")
    
    all_inputs = list(product([0, 1], repeat=8))
    
    correct_count = 0
    total_count = 0
    
    for bits in all_inputs:
        if classical_perm_func:
            expected_output = classical_perm_func(bits)
        else:
            expected_output = bits
        
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i, bit in enumerate(bits):
            if bit == 1:
                circuit.x(qr[i])
        
        build_perm_func(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        
        most_frequent = max(counts, key=counts.get)
        actual_output = tuple(int(most_frequent[7-i]) for i in range(8))
        
        if actual_output == expected_output:
            correct_count += 1
        else:
            print(f"  Input: {bits}")
            print(f"  Expected: {expected_output}")
            print(f"  Actual: {actual_output}")
        
        total_count += 1
    
    accuracy = correct_count / total_count
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.2%}")
    return accuracy

def verify_permutation_inverse(perm_name, build_perm_func, build_inverse_func):
    print(f"Verifying {perm_name} inverse relationship...")
    
    all_inputs = list(product([0, 1], repeat=8))
    sample_size = min(len(all_inputs), 50)
    test_inputs = [all_inputs[i] for i in np.random.choice(len(all_inputs), sample_size, replace=False)]
    
    correct_count = 0
    total_count = 0
    
    for bits in test_inputs:
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i, bit in enumerate(bits):
            if bit == 1:
                circuit.x(qr[i])
        
        build_perm_func(circuit, qr)
        build_inverse_func(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(circuit, simulator)
        job = simulator.run(compiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts(circuit)
        
        most_frequent = max(counts, key=counts.get)
        actual_output = tuple(int(most_frequent[7-i]) for i in range(8))
        
        if actual_output == bits:
            correct_count += 1
        else:
            print(f"  Input: {bits}")
            print(f"  After perm & inverse: {actual_output}")
        
        total_count += 1
    
    accuracy = correct_count / total_count
    print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.2%}")
    return accuracy

def draw_permutation_circuits():
    print("Generating circuit diagrams...")
    
    # Create Initial Permutation circuit
    qr_ip = QuantumRegister(8, 'q')
    cr_ip = ClassicalRegister(8, 'c')
    circuit_ip = QuantumCircuit(qr_ip, cr_ip)
    build_initial_permutation(circuit_ip, qr_ip)
    circuit_ip.measure(qr_ip, cr_ip)
    
    # Create Inverse Initial Permutation circuit
    qr_ip_inv = QuantumRegister(8, 'q')
    cr_ip_inv = ClassicalRegister(8, 'c')
    circuit_ip_inv = QuantumCircuit(qr_ip_inv, cr_ip_inv)
    build_inverse_initial_permutation(circuit_ip_inv, qr_ip_inv)
    circuit_ip_inv.measure(qr_ip_inv, cr_ip_inv)
    
    # Create Final Permutation circuit
    qr_fp = QuantumRegister(8, 'q')
    cr_fp = ClassicalRegister(8, 'c')
    circuit_fp = QuantumCircuit(qr_fp, cr_fp)
    build_final_permutation(circuit_fp, qr_fp)
    circuit_fp.measure(qr_fp, cr_fp)
    
    # Create Inverse Final Permutation circuit
    qr_fp_inv = QuantumRegister(8, 'q')
    cr_fp_inv = ClassicalRegister(8, 'c')
    circuit_fp_inv = QuantumCircuit(qr_fp_inv, cr_fp_inv)
    build_inverse_final_permutation(circuit_fp_inv, qr_fp_inv)
    circuit_fp_inv.measure(qr_fp_inv, cr_fp_inv)
    
    # Draw and save circuits
    circuit_drawer(circuit_ip, filename='initial_permutation.png', output='mpl', style={'name': 'bw'})
    circuit_drawer(circuit_ip_inv, filename='inverse_initial_permutation.png', output='mpl', style={'name': 'bw'})
    circuit_drawer(circuit_fp, filename='final_permutation.png', output='mpl', style={'name': 'bw'})
    circuit_drawer(circuit_fp_inv, filename='inverse_final_permutation.png', output='mpl', style={'name': 'bw'})
    
    print("Circuit diagrams saved as PNG files.")

def compare_implementations():
    print("Testing Initial Permutation against classical implementation...")
    ip_accuracy = verify_permutation("Initial Permutation", build_initial_permutation, classical_initial_permutation)
    
    print("\nTesting Final Permutation against classical implementation...")
    fp_accuracy = verify_permutation("Final Permutation", build_final_permutation, classical_final_permutation)
    
    print("\nTesting Initial Permutation and its inverse...")
    ip_inv_accuracy = verify_permutation_inverse("Initial Permutation", build_initial_permutation, build_inverse_initial_permutation)
    
    print("\nTesting Final Permutation and its inverse...")
    fp_inv_accuracy = verify_permutation_inverse("Final Permutation", build_final_permutation, build_inverse_final_permutation)
    
    print("\nDrawing circuit diagrams...")
    draw_permutation_circuits()
    
    print("\nSummary:")
    print(f"Initial Permutation Accuracy: {ip_accuracy:.2%}")
    print(f"Final Permutation Accuracy: {fp_accuracy:.2%}")
    print(f"Initial Permutation Inverse Accuracy: {ip_inv_accuracy:.2%}")
    print(f"Final Permutation Inverse Accuracy: {fp_inv_accuracy:.2%}")
    print(f"Circuit diagrams have been saved as PNG files in the current directory.")

if __name__ == "__main__":
    compare_implementations()