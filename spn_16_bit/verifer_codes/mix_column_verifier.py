from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def mix_columns(circuit, qubits):
    circuit.cx(qubits[0], qubits[6])
    circuit.cx(qubits[5], qubits[3])
    circuit.cx(qubits[4], qubits[2])
    circuit.cx(qubits[1], qubits[7])
    circuit.cx(qubits[7], qubits[4])
    circuit.cx(qubits[2], qubits[5])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[6], qubits[1])
    return circuit

def inverse_mix_columns(circuit, qubits):
    circuit.cx(qubits[6], qubits[1])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[2], qubits[5])
    circuit.cx(qubits[7], qubits[4])
    circuit.cx(qubits[1], qubits[7])
    circuit.cx(qubits[4], qubits[2])
    circuit.cx(qubits[5], qubits[3])
    circuit.cx(qubits[0], qubits[6])
    return circuit

def optimized_mix_columns(circuit, qubits):
    circuit = mix_columns(circuit, qubits)
    
    # Apply optimal swap sequence found by brute force
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[1], qubits[4])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[4], qubits[6])
    
    return circuit

def optimized_inverse_mix_columns(circuit, qubits):
    # Apply inverse of the optimal swap sequence first (in reverse order)
    circuit.swap(qubits[4], qubits[6])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[1], qubits[4])
    circuit.swap(qubits[0], qubits[2])
    
    circuit = inverse_mix_columns(circuit, qubits)
    return circuit

def classical_mix_columns(state_bits):
    state = [int(bit) for bit in state_bits]
    result = [0] * 8
    result[0] = state[0] ^ state[6]
    result[1] = state[1] ^ state[4] ^ state[7]
    result[2] = state[2] ^ state[4] ^ state[5]
    result[3] = state[3] ^ state[5]
    result[4] = state[2] ^ state[4]
    result[5] = state[0] ^ state[3] ^ state[5]
    result[6] = state[0] ^ state[1] ^ state[6]
    result[7] = state[1] ^ state[7]
    return ''.join(str(bit) for bit in result)

def verify_mix_columns():
    test_cases = [
        "00000000", "11111111", "10101010", "01010101",
        "00001111", "11110000", "00110011", "11001100"
    ]
    
    print("Verifying optimized Mix Columns:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        expected_output = classical_mix_columns(input_bits)
        
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(8):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = optimized_mix_columns(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        output = max(counts, key=counts.get)
        output_ordered = ''.join(reversed(output))
        
        if output_ordered == expected_output:
            correct_count += 1
            print(f"✓ Input: {input_bits} -> Output: {output_ordered} - Expected: {expected_output}")
        else:
            print(f"✗ Input: {input_bits} -> Output: {output_ordered} - Expected: {expected_output}")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nResults: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def verify_inverse_property():
    test_cases = [
        "00000000", "11111111", "10101010", "01010101",
        "00001111", "11110000", "00110011", "11001100"
    ]
    
    print("\nVerifying Inverse Property:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(8):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = optimized_mix_columns(circuit, qr)
        circuit = optimized_inverse_mix_columns(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        output = max(counts, key=counts.get)
        output_ordered = ''.join(reversed(output))
        
        if output_ordered == input_bits:
            correct_count += 1
            print(f"✓ Input: {input_bits} -> Output after inverse: {output_ordered}")
        else:
            print(f"✗ Input: {input_bits} -> Output after inverse: {output_ordered}")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nInverse Property Results: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def draw_optimized_circuit():
    from qiskit.visualization import circuit_drawer
    import matplotlib.pyplot as plt
    
    qr = QuantumRegister(8, 'q')
    circuit = QuantumCircuit(qr)
    
    circuit = optimized_mix_columns(circuit, qr)
    
    fig = plt.figure(figsize=(12, 8))
    circuit_drawing = circuit_drawer(circuit, output='mpl', 
                                    style={'name': 'bw'})
    
    plt.title("Optimized Mix Columns Circuit", fontsize=16)
    plt.savefig('optimized_mix_columns_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Optimized Mix Columns circuit drawing saved as 'optimized_mix_columns_circuit.png'")

if __name__ == "__main__":
    verify_mix_columns()
    verify_inverse_property()
    draw_optimized_circuit()