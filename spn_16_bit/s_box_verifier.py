from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np

def s_box_4bit(circuit, qubits):
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
    return circuit

def verify_s_box_implementation():
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
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
        
        circuit = s_box_4bit(circuit, qr)
        
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

def draw_sbox_circuit():
    qr = QuantumRegister(4, 'q')
    circuit = QuantumCircuit(qr)
    
    circuit = s_box_4bit(circuit, qr)
    
    fig = plt.figure(figsize=(12, 8))
    circuit_drawing = circuit_drawer(circuit, output='mpl', 
                                    style={'name': 'bw'})
    
    plt.title("S-box Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('sbox_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Circuit drawing saved as 'sbox_circuit.png'")

if __name__ == "__main__":
    print("Verifying S-box implementation against lookup table:")
    verify_s_box_implementation()
    
    print("\nGenerating S-box circuit diagram:")
    draw_sbox_circuit()