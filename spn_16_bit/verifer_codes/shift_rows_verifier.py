from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np

def shift_rows(circuit, qubits):
    circuit.swap(qubits[4], qubits[12])
    circuit.swap(qubits[5], qubits[13])
    circuit.swap(qubits[6], qubits[14])
    circuit.swap(qubits[7], qubits[15])
    return circuit

def inverse_shift_rows(circuit, qubits):
    circuit.swap(qubits[7], qubits[15])
    circuit.swap(qubits[6], qubits[14])
    circuit.swap(qubits[5], qubits[13])
    circuit.swap(qubits[4], qubits[12])
    return circuit

def classical_shift_rows(state_bits):
    state = list(state_bits)
    
    state[4], state[12] = state[12], state[4]
    state[5], state[13] = state[13], state[5]
    state[6], state[14] = state[14], state[6]
    state[7], state[15] = state[15], state[7]
    
    return ''.join(state)

def classical_inverse_shift_rows(state_bits):
    return classical_shift_rows(state_bits)

def verify_shift_rows():
    test_cases = [
        "0000000000000000", 
        "1111111111111111", 
        "1010101010101010", 
        "0101010101010101",
        "0000111100001111", 
        "1111000011110000", 
        "0011001100110011", 
        "1100110011001100"
    ]
    
    print("Verifying Shift Rows implementation:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        expected_output = classical_shift_rows(input_bits)
        
        qr = QuantumRegister(16, 'q')
        cr = ClassicalRegister(16, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(16):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = shift_rows(circuit, qr)
        
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
        "0000000000000000", 
        "1111111111111111", 
        "1010101010101010", 
        "0101010101010101",
        "0001001000110100"
    ]
    
    print("\nVerifying Inverse Property:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        if len(input_bits) != 16:
            input_bits = input_bits[:16].zfill(16)
            print(f"Warning: Adjusted input to 16 bits: {input_bits}")
        
        qr = QuantumRegister(16, 'q')
        cr = ClassicalRegister(16, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(16):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = shift_rows(circuit, qr)
        circuit = inverse_shift_rows(circuit, qr)
        
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
            print(f"✗ Input: {input_bits} -> Output after inverse: {output_ordered} (should match input)")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nInverse Property Results: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

if __name__ == "__main__":
    verify_shift_rows()
    verify_inverse_property()