from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import itertools
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

def mix_columns_with_swaps(circuit, qubits, swaps):
    circuit = mix_columns(circuit, qubits)
    for i, j in swaps:
        circuit.swap(qubits[i], qubits[j])
    return circuit

def test_swap_combination(swaps, test_cases):
    correct_count = 0
    for input_bits in test_cases:
        expected_output = classical_mix_columns(input_bits)
        
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(8):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = mix_columns_with_swaps(circuit, qr, swaps)
        
        circuit.measure(qr, cr)
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        output = list(counts.keys())[0]
        output_ordered = ''.join(reversed(output))
        
        if output_ordered == expected_output:
            correct_count += 1
    
    return correct_count

def find_optimal_swaps():
    test_cases = [
        "00000000", "11111111", "10101010", "01010101",
        "00001111", "11110000", "00110011", "11001100"
    ]
    
    best_swaps = []
    best_score = 0
    
    print("Testing swap combinations with 1 swap...")
    
    all_possible_swaps = list(itertools.combinations(range(8), 2))
    
    for num_swaps in range(1, 5):
        for swaps in itertools.combinations(all_possible_swaps, num_swaps):
            score = test_swap_combination(swaps, test_cases)
            if score > best_score:
                best_score = score
                best_swaps = swaps
                print(f"New best found: {swaps} with score {score}/{len(test_cases)}")
                if score == len(test_cases):
                    print("Perfect match found!")
                    return best_swaps
        
        print(f"Best after {num_swaps} swaps: {best_swaps} with score {best_score}/{len(test_cases)}")
    
    return best_swaps

def verify_with_swaps(swaps):
    test_cases = [
        "00000000", "11111111", "10101010", "01010101",
        "00001111", "11110000", "00110011", "11001100"
    ]
    
    print("\nVerifying Mix Columns with optimal swaps:")
    correct_count = 0
    
    for input_bits in test_cases:
        expected_output = classical_mix_columns(input_bits)
        
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(8):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        circuit = mix_columns_with_swaps(circuit, qr, swaps)
        
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
    
    accuracy = (correct_count / len(test_cases)) * 100
    print(f"\nResults: {correct_count}/{len(test_cases)} correct ({accuracy:.2f}%)")

if __name__ == "__main__":
    optimal_swaps = find_optimal_swaps()
    print(f"\nOptimal swap sequence: {optimal_swaps}")
    verify_with_swaps(optimal_swaps)