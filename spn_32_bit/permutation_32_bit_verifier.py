from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def permutation_32bit(circuit, qubits):
    for i in range(4):
        byte_start = i * 8
        for j in range(i + 1):
            for k in range(7):
                circuit.swap(qubits[byte_start + k], qubits[byte_start + k + 1])
    return circuit

def inverse_permutation_32bit(circuit, qubits):
    for i in range(3, -1, -1):
        byte_start = i * 8
        for j in range(i + 1):
            for k in range(7, 0, -1):
                circuit.swap(qubits[byte_start + k], qubits[byte_start + k - 1])
    return circuit

def classical_permutation_32bit(input_bits):
    bits = list(input_bits)
    for i in range(4):
        byte_start = i * 8
        for j in range(i + 1):
            for k in range(7):
                bits[byte_start + k], bits[byte_start + k + 1] = bits[byte_start + k + 1], bits[byte_start + k]
    return ''.join(bits)

def classical_inverse_permutation_32bit(input_bits):
    bits = list(input_bits)
    for i in range(3, -1, -1):
        byte_start = i * 8
        for j in range(i + 1):
            for k in range(7, 0, -1):
                bits[byte_start + k], bits[byte_start + k - 1] = bits[byte_start + k - 1], bits[byte_start + k]
    return ''.join(bits)

def verify_permutation():
    test_cases = [
        "00000000000000000000000000000000",
        "11111111111111111111111111111111",
        "10101010101010101010101010101010",
        "01010101010101010101010101010101",
        "00001111000011110000111100001111",
        "11110000111100001111000011110000",
        "00110011001100110011001100110011",
        "11001100110011001100110011001100"
    ]
    
    print("Verifying 32-bit Permutation:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        expected_output = classical_permutation_32bit(input_bits)
        
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(32):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = permutation_32bit(circuit, qr)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        output = max(counts, key=counts.get)
        output_ordered = ''.join(reversed(output))
        
        if output_ordered == expected_output:
            correct_count += 1
            print(f"✓ Input: {input_bits} -> Output matches expected")
        else:
            print(f"✗ Input: {input_bits} -> Output: {output_ordered} - Expected: {expected_output}")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nResults: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def verify_inverse_property():
    test_cases = [
        "00000000000000000000000000000000",
        "11111111111111111111111111111111",
        "10101010101010101010101010101010",
        "01010101010101010101010101010101",
        "00001111000011110000111100001111"
    ]
    
    print("\nVerifying Inverse Property:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(32):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = permutation_32bit(circuit, qr)
        circuit = inverse_permutation_32bit(circuit, qr)
        
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

def analyze_permutation_effect():
    print("\nAnalyzing permutation effect on individual bits:")
    
    for bit_pos in range(32):
        input_bits = ['0'] * 32
        input_bits[bit_pos] = '1'
        input_str = ''.join(input_bits)
        
        output_str = classical_permutation_32bit(input_str)
        ones_positions = [i for i, bit in enumerate(output_str) if bit == '1']
        
        print(f"Bit {bit_pos} maps to positions: {ones_positions}")

if __name__ == "__main__":
    verify_permutation()
    verify_inverse_property()
    analyze_permutation_effect()