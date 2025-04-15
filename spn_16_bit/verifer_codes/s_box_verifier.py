from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np

def fully_quantum_s_box(circuit, qubits):
    circuit.swap(qubits[1], qubits[2])
    circuit.swap(qubits[0], qubits[3])
    
    circuit.cx(qubits[2], qubits[1])
    circuit.ccx(qubits[3], qubits[1], qubits[0])
    circuit.ccx(qubits[0], qubits[2], qubits[3])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.mcx([qubits[0], qubits[1], qubits[3]], qubits[2])
    circuit.cx(qubits[3], qubits[0])
    circuit.x(qubits[2])
    circuit.x(qubits[3])
    circuit.cx(qubits[0], qubits[2])
    circuit.ccx(qubits[2], qubits[1], qubits[0])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.ccx(qubits[0], qubits[1], qubits[3])
    
    circuit.swap(qubits[0], qubits[1])
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[1], qubits[2])
    
    return circuit

def inverse_fully_quantum_s_box(circuit, qubits):
    circuit.swap(qubits[1], qubits[2])
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[0], qubits[1])
    
    circuit.ccx(qubits[0], qubits[1], qubits[3])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.ccx(qubits[2], qubits[1], qubits[0])
    circuit.cx(qubits[0], qubits[2])
    circuit.x(qubits[3])
    circuit.x(qubits[2])
    circuit.cx(qubits[3], qubits[0])
    circuit.mcx([qubits[0], qubits[1], qubits[3]], qubits[2])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.ccx(qubits[0], qubits[2], qubits[3])
    circuit.ccx(qubits[3], qubits[1], qubits[0])
    circuit.cx(qubits[2], qubits[1])
    
    circuit.swap(qubits[0], qubits[3])
    circuit.swap(qubits[1], qubits[2])
    
    return circuit

def verify_fully_quantum_s_box():
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    print("Verifying fully quantum S-box implementation:")
    correct_count = 0
    total_tests = 16
    
    for input_val in range(16):
        expected_output = s_box_table[input_val]
        
        input_binary = format(input_val, '04b')
        expected_binary = format(expected_output, '04b')
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        circuit = fully_quantum_s_box(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        output = max(counts, key=counts.get)
        output_val = int(output, 2)
        output_binary = format(output_val, '04b')
        
        if output_val == expected_output:
            correct_count += 1
            print(f"✓ Input: {input_binary} ({input_val}) -> Output: {output_binary} ({output_val}) - Expected: {expected_binary} ({expected_output})")
        else:
            print(f"✗ Input: {input_binary} ({input_val}) -> Output: {output_binary} ({output_val}) - Expected: {expected_binary} ({expected_output})")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nResults: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def verify_inverse_property():
    print("\nVerifying inverse property of fully quantum S-box:")
    correct_count = 0
    total_tests = 16
    
    for input_val in range(16):
        input_binary = format(input_val, '04b')
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        circuit = fully_quantum_s_box(circuit, qr)
        circuit = inverse_fully_quantum_s_box(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        output = max(counts, key=counts.get)
        output_val = int(output, 2)
        output_binary = format(output_val, '04b')
        
        if output_val == input_val:
            correct_count += 1
            print(f"✓ Input: {input_binary} ({input_val}) -> Output after inverse: {output_binary} ({output_val})")
        else:
            print(f"✗ Input: {input_binary} ({input_val}) -> Output after inverse: {output_binary} ({output_val})")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nInverse Property Results: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def draw_quantum_sbox_circuit():
    qr = QuantumRegister(4, 'q')
    circuit = QuantumCircuit(qr)
    
    circuit = fully_quantum_s_box(circuit, qr)
    
    fig = plt.figure(figsize=(14, 8))
    circuit_drawing = circuit_drawer(circuit, output='mpl', style={'name': 'bw'})
    
    plt.title("Fully Quantum S-box Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('fully_quantum_sbox_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Fully quantum S-box circuit drawing saved as 'fully_quantum_sbox_circuit.png'")
    
    qr_inv = QuantumRegister(4, 'q')
    circuit_inv = QuantumCircuit(qr_inv)
    
    circuit_inv = inverse_fully_quantum_s_box(circuit_inv, qr_inv)
    
    fig = plt.figure(figsize=(14, 8))
    circuit_drawing_inv = circuit_drawer(circuit_inv, output='mpl', style={'name': 'bw'})
    
    plt.title("Inverse Fully Quantum S-box Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('inverse_fully_quantum_sbox_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Inverse fully quantum S-box circuit drawing saved as 'inverse_fully_quantum_sbox_circuit.png'")

if __name__ == "__main__":
    print("Testing fully quantum S-box implementation:")
    verify_fully_quantum_s_box()
    verify_inverse_property()
    draw_quantum_sbox_circuit()