from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def mix_columns(circuit, qubits):
    # Mix 4 columns, each column has 8 bits (2 from each row)
    for col in range(4):
        col_start = col * 2
        
        # Mix first row with second
        circuit.cx(qubits[col_start], qubits[8+col_start])
        circuit.cx(qubits[col_start+1], qubits[8+col_start+1])
        
        # Mix second row with third
        circuit.cx(qubits[8+col_start], qubits[16+col_start])
        circuit.cx(qubits[8+col_start+1], qubits[16+col_start+1])
        
        # Mix third row with fourth
        circuit.cx(qubits[16+col_start], qubits[24+col_start])
        circuit.cx(qubits[16+col_start+1], qubits[24+col_start+1])
        
        # Mix fourth row with first (forming a cycle)
        circuit.cx(qubits[24+col_start], qubits[col_start])
        circuit.cx(qubits[24+col_start+1], qubits[col_start+1])
        
        # Additional mixing for diffusion
        circuit.cx(qubits[col_start], qubits[16+col_start])
        circuit.cx(qubits[8+col_start], qubits[24+col_start])
    
    return circuit

def inverse_mix_columns(circuit, qubits):
    # Inverse of mix_columns (operations in reverse order)
    for col in range(3, -1, -1):
        col_start = col * 2
        
        # Undo additional mixing
        circuit.cx(qubits[8+col_start], qubits[24+col_start])
        circuit.cx(qubits[col_start], qubits[16+col_start])
        
        # Undo cyclic mixing
        circuit.cx(qubits[24+col_start+1], qubits[col_start+1])
        circuit.cx(qubits[24+col_start], qubits[col_start])
        circuit.cx(qubits[16+col_start+1], qubits[24+col_start+1])
        circuit.cx(qubits[16+col_start], qubits[24+col_start])
        circuit.cx(qubits[8+col_start+1], qubits[16+col_start+1])
        circuit.cx(qubits[8+col_start], qubits[16+col_start])
        circuit.cx(qubits[col_start+1], qubits[8+col_start+1])
        circuit.cx(qubits[col_start], qubits[8+col_start])
    
    return circuit

def corrected_classical_mix_columns(input_bits):
    # Convert input string to list of bit values
    bits = [int(bit) for bit in input_bits]
    result = bits.copy()
    
    # Apply mix columns to each column
    for col in range(4):
        col_start = col * 2
        
        # Store original values to avoid overwriting during XOR operations
        original = bits.copy()
        
        # Mix first row with second
        result[8+col_start] = result[8+col_start] ^ original[col_start]
        result[8+col_start+1] = result[8+col_start+1] ^ original[col_start+1]
        
        # Mix second row with third
        result[16+col_start] = result[16+col_start] ^ original[8+col_start]
        result[16+col_start+1] = result[16+col_start+1] ^ original[8+col_start+1]
        
        # Mix third row with fourth
        result[24+col_start] = result[24+col_start] ^ original[16+col_start]
        result[24+col_start+1] = result[24+col_start+1] ^ original[16+col_start+1]
        
        # Mix fourth row with first (forming a cycle)
        result[col_start] = result[col_start] ^ original[24+col_start]
        result[col_start+1] = result[col_start+1] ^ original[24+col_start+1]
        
        # Additional mixing for diffusion
        result[16+col_start] = result[16+col_start] ^ original[col_start]
        result[24+col_start] = result[24+col_start] ^ original[8+col_start]
    
    # Convert back to string
    return ''.join(str(bit) for bit in result)

def verify_mix_columns():
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
    
    print("Verifying MixColumns implementation:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        # Calculate expected output using corrected classical function
        expected_output = corrected_classical_mix_columns(input_bits)
        
        # Create quantum circuit
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input state
        for i in range(32):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        # Apply Mix Columns
        circuit = mix_columns(circuit, qr)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        # Get most frequent result
        output = max(counts, key=counts.get)
        output_ordered = ''.join(reversed(output))
        
        # Compare with expected
        if output_ordered == expected_output:
            correct_count += 1
            print(f"✓ Input: {input_bits[:8]}... -> Output matches expected")
        else:
            print(f"✗ Input: {input_bits[:8]}... -> Output doesn't match")
            print(f"  Got:      {output_ordered[:16]}...")
            print(f"  Expected: {expected_output[:16]}...")
            
            # Debug: print detailed comparison
            print("  Detailed comparison (first 16 bits):")
            for i in range(16):
                match = "✓" if output_ordered[i] == expected_output[i] else "✗"
                print(f"    Bit {i}: Got {output_ordered[i]}, Expected {expected_output[i]} {match}")
    
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
    
    print("\nVerifying MixColumns Inverse Property:")
    correct_count = 0
    total_tests = len(test_cases)
    
    for input_bits in test_cases:
        # Create quantum circuit
        qr = QuantumRegister(32, 'q')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input state
        for i in range(32):
            if input_bits[i] == '1':
                circuit.x(qr[i])
        
        # Apply Mix Columns followed by Inverse Mix Columns
        circuit = mix_columns(circuit, qr)
        circuit = inverse_mix_columns(circuit, qr)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=10000)
        result = job.result()
        counts = result.get_counts()
        
        # Get most frequent result
        output = max(counts, key=counts.get)
        output_ordered = ''.join(reversed(output))
        
        # Check if output matches input (should be identity operation)
        if output_ordered == input_bits:
            correct_count += 1
            print(f"✓ Input: {input_bits[:8]}... -> Output after inverse matches input")
        else:
            print(f"✗ Input: {input_bits[:8]}... -> Output after inverse doesn't match input")
            print(f"  Got:      {output_ordered[:16]}...")
            print(f"  Expected: {input_bits[:16]}...")
    
    accuracy = (correct_count / total_tests) * 100
    print(f"\nInverse Property Results: {correct_count}/{total_tests} correct ({accuracy:.2f}%)")
    
    return correct_count, total_tests, accuracy

def analyze_diffusion():
    print("\nAnalyzing diffusion properties of MixColumns:")
    
    # Test with inputs that have only a single bit set
    for bit_pos in range(0, 32, 4):  # Sample positions
        input_bits = ['0'] * 32
        input_bits[bit_pos] = '1'
        input_str = ''.join(input_bits)
        
        output_str = corrected_classical_mix_columns(input_str)
        
        # Count number of bits affected
        bits_changed = sum(1 for i in range(32) if output_str[i] == '1')
        
        print(f"Setting bit {bit_pos} affects {bits_changed} bits in the output")
        
        # Show which bits were affected
        affected_positions = [i for i in range(32) if output_str[i] == '1']
        print(f"  Affected positions: {affected_positions}")

def draw_mixcolumns_circuit():
    # Create circuit for one column only (for clarity)
    qr = QuantumRegister(32, 'q')
    circuit = QuantumCircuit(qr)
    
    # Just show one column operation
    col = 0
    col_start = col * 2
    
    # Mix first row with second
    circuit.cx(qr[col_start], qr[8+col_start])
    circuit.cx(qr[col_start+1], qr[8+col_start+1])
    
    # Mix second row with third
    circuit.cx(qr[8+col_start], qr[16+col_start])
    circuit.cx(qr[8+col_start+1], qr[16+col_start+1])
    
    # Mix third row with fourth
    circuit.cx(qr[16+col_start], qr[24+col_start])
    circuit.cx(qr[16+col_start+1], qr[24+col_start+1])
    
    # Mix fourth row with first
    circuit.cx(qr[24+col_start], qr[col_start])
    circuit.cx(qr[24+col_start+1], qr[col_start+1])
    
    # Additional mixing for diffusion
    circuit.cx(qr[col_start], qr[16+col_start])
    circuit.cx(qr[8+col_start], qr[24+col_start])
    
    # Draw the circuit
    circuit_drawing = circuit_drawer(circuit, output='mpl', style={'name': 'bw'})
    plt.title("MixColumns Circuit (Single Column)", fontsize=16)
    plt.tight_layout()
    plt.savefig('mixcolumns_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("MixColumns circuit diagram saved as 'mixcolumns_circuit.png'")
    
    # Draw full circuit (all columns)
    qr_full = QuantumRegister(32, 'q')
    circuit_full = QuantumCircuit(qr_full)
    circuit_full = mix_columns(circuit_full, qr_full)
    
    circuit_drawing = circuit_drawer(circuit_full, output='mpl', style={'name': 'bw'})
    plt.title("MixColumns Full Circuit", fontsize=16)
    plt.tight_layout()
    plt.savefig('mixcolumns_full_circuit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Full MixColumns circuit diagram saved as 'mixcolumns_full_circuit.png'")

if __name__ == "__main__":
    verify_mix_columns()
    verify_inverse_property()
    analyze_diffusion()
    draw_mixcolumns_circuit()