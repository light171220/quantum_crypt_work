from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def optimized_s_box_4bit(circuit, qubits):
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

def inverse_optimized_s_box_4bit(circuit, qubits):
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

def verify_s_box():
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    print("Verifying S-box implementation:")
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
        
        circuit = optimized_s_box_4bit(circuit, qr)
        
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
    print("\nVerifying S-box Inverse Property:")
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
        
        circuit = optimized_s_box_4bit(circuit, qr)
        circuit = inverse_optimized_s_box_4bit(circuit, qr)
        
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

def draw_sbox_circuits():
    # Draw S-box circuit
    qr = QuantumRegister(4, 'q')
    sbox_circuit = QuantumCircuit(qr)
    sbox_circuit = optimized_s_box_4bit(sbox_circuit, qr)
    
    fig = plt.figure(figsize=(14, 10))
    circuit_drawing = circuit_drawer(sbox_circuit, output='mpl', style={'name': 'bw'})
    plt.title("Optimized S-box Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('sbox_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Draw inverse S-box circuit
    qr_inv = QuantumRegister(4, 'q')
    inv_sbox_circuit = QuantumCircuit(qr_inv)
    inv_sbox_circuit = inverse_optimized_s_box_4bit(inv_sbox_circuit, qr_inv)
    
    fig = plt.figure(figsize=(14, 10))
    inv_circuit_drawing = circuit_drawer(inv_sbox_circuit, output='mpl', style={'name': 'bw'})
    plt.title("Inverse Optimized S-box Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('inverse_sbox_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Circuit diagrams saved as 'sbox_circuit.png' and 'inverse_sbox_circuit.png'")

if __name__ == "__main__":
    verify_s_box()
    verify_inverse_property()
    draw_sbox_circuits()