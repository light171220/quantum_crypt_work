from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def shift_rows(circuit, qubits):
    # Organize 32 bits as 4 rows of 8 bits (similar to AES organization)
    # Row 0: No shift (qubits 0-7)
    # Row 1: Shift left by 2 bits (qubits 8-15)
    circuit.swap(qubits[8], qubits[10])
    circuit.swap(qubits[9], qubits[11])
    circuit.swap(qubits[12], qubits[14])
    circuit.swap(qubits[13], qubits[15])
    
    # Row 2: Shift left by 4 bits (qubits 16-23)
    circuit.swap(qubits[16], qubits[20])
    circuit.swap(qubits[17], qubits[21])
    circuit.swap(qubits[18], qubits[22])
    circuit.swap(qubits[19], qubits[23])
    
    # Row 3: Shift left by 6 bits (qubits 24-31)
    circuit.swap(qubits[24], qubits[30])
    circuit.swap(qubits[25], qubits[31])
    circuit.swap(qubits[26], qubits[28])
    circuit.swap(qubits[27], qubits[29])
    
    return circuit

def inverse_shift_rows(circuit, qubits):
    # Reverse of shift_rows
    # Row 3: Shift right by 6 bits (qubits 24-31)
    circuit.swap(qubits[27], qubits[29])
    circuit.swap(qubits[26], qubits[28])
    circuit.swap(qubits[25], qubits[31])
    circuit.swap(qubits[24], qubits[30])
    
    # Row 2: Shift right by 4 bits (qubits 16-23)
    circuit.swap(qubits[19], qubits[23])
    circuit.swap(qubits[18], qubits[22])
    circuit.swap(qubits[17], qubits[21])
    circuit.swap(qubits[16], qubits[20])
    
    # Row 1: Shift right by 2 bits (qubits 8-15)
    circuit.swap(qubits[13], qubits[15])
    circuit.swap(qubits[12], qubits[14])
    circuit.swap(qubits[9], qubits[11])
    circuit.swap(qubits[8], qubits[10])
    
    return circuit

def classical_shift_rows(input_bits):
    bits = list(input_bits)
    
    # Row 1: Shift left by 2 bits (qubits 8-15)
    bits[8], bits[10] = bits[10], bits[8]
    bits[9], bits[11] = bits[11], bits[9]
    bits[12], bits[14] = bits[14], bits[12]
    bits[13], bits[15] = bits[15], bits[13]
    
    # Row 2: Shift left by 4 bits (qubits 16-23)
    bits[16], bits[20] = bits[20], bits[16]
    bits[17], bits[21] = bits[21], bits[17]
    bits[18], bits[22] = bits[22], bits[18]
    bits[19], bits[23] = bits[23], bits[19]
    
    # Row 3: Shift left by 6 bits (qubits 24-31)
    bits[24], bits[30] = bits[30], bits[24]
    bits[25], bits[31] = bits[31], bits[25]
    bits[26], bits[28] = bits[28], bits[26]
    bits[27], bits[29] = bits[29], bits[27]
    
    return ''.join(bits)

def verify_shift_rows():
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
    
    print("Verifying ShiftRows implementation:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        expected_output = classical_shift_rows(input_bits)
        
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(32):
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
            print(f"✓ Input: {input_bits[:8]}... -> Output matches expected")
        else:
            print(f"✗ Input: {input_bits[:8]}... -> Output doesn't match")
            print(f"  Got:      {output_ordered[:8]}...")
            print(f"  Expected: {expected_output[:8]}...")
    
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
    
    print("\nVerifying ShiftRows Inverse Property:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        for i in range(32):
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
            print(f"✓ Input: {input_bits[:8]}... -> Output after inverse matches input")
        else:
            print(f"✗ Input: {input_bits[:8]}... -> Output after inverse: {output_ordered[:8]}...")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nInverse Property Results: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def draw_shiftrows_circuits():
    # Draw ShiftRows circuit
    qr = QuantumRegister(32, 'q')
    shift_circuit = QuantumCircuit(qr)
    shift_circuit = shift_rows(shift_circuit, qr)
    
    fig = plt.figure(figsize=(14, 10))
    circuit_drawing = circuit_drawer(shift_circuit, output='mpl', style={'name': 'bw'})
    plt.title("ShiftRows Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('shiftrows_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Draw inverse ShiftRows circuit
    qr_inv = QuantumRegister(32, 'q')
    inv_shift_circuit = QuantumCircuit(qr_inv)
    inv_shift_circuit = inverse_shift_rows(inv_shift_circuit, qr_inv)
    
    fig = plt.figure(figsize=(14, 10))
    inv_circuit_drawing = circuit_drawer(inv_shift_circuit, output='mpl', style={'name': 'bw'})
    plt.title("Inverse ShiftRows Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('inverse_shiftrows_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Circuit diagrams saved as 'shiftrows_circuit.png' and 'inverse_shiftrows_circuit.png'")

if __name__ == "__main__":
    verify_shift_rows()
    verify_inverse_property()
    draw_shiftrows_circuits()