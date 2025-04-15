from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

def s_box_with_swaps(circuit, qubits, swaps):
    circuit = s_box_4bit(circuit, qubits)
    for i, j in swaps:
        circuit.swap(qubits[i], qubits[j])
    return circuit

def s_box_with_swaps_and_bit_flips(circuit, qubits, swaps, bit_flips):
    circuit = s_box_4bit(circuit, qubits)
    
    # Apply swaps
    for i, j in swaps:
        circuit.swap(qubits[i], qubits[j])
    
    # Apply bit flips (NOT gates)
    for i in bit_flips:
        circuit.x(qubits[i])
    
    return circuit

def get_sbox_mapping():
    actual_outputs = {}
    for input_val in range(16):
        input_binary = format(input_val, '04b')
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        # Apply S-box
        circuit = s_box_4bit(circuit, qr)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        output = list(counts.keys())[0]
        output_val = int(output, 2)
        actual_outputs[input_val] = output_val
    
    return actual_outputs

def test_transformation(swaps=[], bit_flips=[], input_reordering=None, output_reordering=None):
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    correct_count = 0
    results = {}
    
    for input_val in range(16):
        # Apply input reordering if specified
        if input_reordering:
            # Reorder the input bits according to the mapping
            input_bits = list(format(input_val, '04b').zfill(4))
            reordered_input_bits = [input_bits[i] for i in input_reordering]
            actual_input = int(''.join(reordered_input_bits), 2)
        else:
            actual_input = input_val
        
        expected_output = s_box_table[input_val]
        
        # Create circuit
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input
        input_binary = format(actual_input, '04b')
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        # Apply S-box with transformations
        circuit = s_box_with_swaps_and_bit_flips(circuit, qr, swaps, bit_flips)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        output = list(counts.keys())[0]
        output_val = int(output, 2)
        
        # Apply output reordering if specified
        if output_reordering:
            output_bits = list(format(output_val, '04b').zfill(4))
            reordered_output_bits = [output_bits[i] for i in output_reordering]
            output_val = int(''.join(reordered_output_bits), 2)
        
        results[input_val] = output_val
        
        if output_val == expected_output:
            correct_count += 1
    
    return correct_count, results

def find_optimal_transformation():
    best_score = 0
    best_config = None
    
    # Get the current S-box mapping
    actual_mapping = get_sbox_mapping()
    
    print("Current S-box mapping:")
    for i in range(16):
        print(f"{i:2d} -> {actual_mapping[i]:2d}")
    
    # Try all permutations of input and output bits
    all_perms = list(itertools.permutations(range(4)))
    
    # Try combinations of swaps, bit flips, and reorderings
    for num_swaps in range(4):
        for swaps in itertools.combinations(itertools.combinations(range(4), 2), num_swaps):
            swaps = tuple(swaps) if num_swaps > 0 else []
            
            for num_flips in range(5):  # 0 to 4 bit flips
                for bit_flips in itertools.combinations(range(4), num_flips):
                    for input_perm in [None] + all_perms[:5]:  # Try None and a few perms
                        for output_perm in [None] + all_perms[:5]:  # Try None and a few perms
                            score, mapping = test_transformation(swaps, bit_flips, input_perm, output_perm)
                            
                            if score > best_score:
                                best_score = score
                                best_config = {
                                    "swaps": swaps,
                                    "bit_flips": bit_flips,
                                    "input_reordering": input_perm,
                                    "output_reordering": output_perm,
                                    "score": score,
                                    "mapping": mapping
                                }
                                
                                print(f"New best found: score {score}/16")
                                print(f"  Swaps: {swaps}")
                                print(f"  Bit flips: {bit_flips}")
                                print(f"  Input reordering: {input_perm}")
                                print(f"  Output reordering: {output_perm}")
                                
                                if score == 16:
                                    print("Perfect match found!")
                                    return best_config
    
    print("\nBest configuration:")
    print(f"  Score: {best_config['score']}/16")
    print(f"  Swaps: {best_config['swaps']}")
    print(f"  Bit flips: {best_config['bit_flips']}")
    print(f"  Input reordering: {best_config['input_reordering']}")
    print(f"  Output reordering: {best_config['output_reordering']}")
    
    return best_config

def try_alternative_sbox_implementations():
    """Try alternative S-box implementations to see if they work better"""
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    # First implementation from the LIGHTER-R paper
    def s_box_lighter_r(circuit, qubits):
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[3])
        circuit.cx(qubits[1], qubits[2])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[3], qubits[2])
        circuit.x(qubits[3])
        circuit.cx(qubits[1], qubits[3])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[0], qubits[2])
        circuit.cx(qubits[3], qubits[0])
        circuit.x(qubits[2])
        circuit.cx(qubits[0], qubits[3])
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[1], qubits[2])
        return circuit
    
    # Another implementation directly using the lookup table
    def s_box_direct(circuit, qubits):
        # Implementation based on the truth table directly
        # We'll just use a simplified version that maps a few values correctly
        
        # X gates on specific combinations of inputs
        circuit.x(qubits[0])
        circuit.x(qubits[1])
        
        # Use some control gates to implement the mapping
        circuit.cx(qubits[0], qubits[2])
        circuit.cx(qubits[1], qubits[3])
        circuit.ccx(qubits[0], qubits[1], qubits[2])
        circuit.ccx(qubits[2], qubits[3], qubits[0])
        
        # More transformations
        circuit.cx(qubits[3], qubits[1])
        circuit.x(qubits[2])
        
        return circuit
    
    # Test both implementations
    implementations = [
        ("Original", s_box_4bit),
        ("LIGHTER-R", s_box_lighter_r),
        ("Direct", s_box_direct)
    ]
    
    best_impl = None
    best_score = 0
    
    for name, impl in implementations:
        correct_count = 0
        results = {}
        
        for input_val in range(16):
            expected_output = s_box_table[input_val]
            
            qr = QuantumRegister(4, 'q')
            cr = ClassicalRegister(4, 'c')
            circuit = QuantumCircuit(qr, cr)
            
            # Set input
            input_binary = format(input_val, '04b')
            for i in range(4):
                if input_binary[3-i] == '1':
                    circuit.x(qr[i])
            
            # Apply S-box
            circuit = impl(circuit, qr)
            
            # Measure
            circuit.measure(qr, cr)
            
            # Simulate
            simulator = AerSimulator()
            job = simulator.run(circuit, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            output = list(counts.keys())[0]
            output_val = int(output, 2)
            results[input_val] = output_val
            
            if output_val == expected_output:
                correct_count += 1
        
        print(f"{name} implementation score: {correct_count}/16")
        
        if correct_count > best_score:
            best_score = correct_count
            best_impl = name
    
    print(f"Best implementation: {best_impl} with score {best_score}/16")

def try_custom_sbox_implementation():
    """Try to create a custom S-box implementation that works correctly"""
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    def custom_sbox(circuit, qubits):
        # Input bits are in reverse order (q[0] is MSB)
        # Perform a simplified direct mapping to demonstrate
        
        # First, we'll try to optimize for a few specific mappings
        # This is a simplified version - a real implementation would be more complex
        
        # Example gates that might help implement the S-box
        circuit.x(qubits[1])
        circuit.x(qubits[2])
        
        circuit.cx(qubits[0], qubits[3])
        circuit.cx(qubits[1], qubits[2])
        
        circuit.ccx(qubits[0], qubits[1], qubits[3])
        circuit.ccx(qubits[2], qubits[3], qubits[0])
        
        circuit.cx(qubits[0], qubits[1])
        circuit.cx(qubits[2], qubits[3])
        
        circuit.x(qubits[0])
        
        return circuit
    
    correct_count = 0
    results = {}
    
    for input_val in range(16):
        expected_output = s_box_table[input_val]
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input
        input_binary = format(input_val, '04b')
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        # Apply custom S-box
        circuit = custom_sbox(circuit, qr)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        output = list(counts.keys())[0]
        output_val = int(output, 2)
        results[input_val] = output_val
        
        if output_val == expected_output:
            correct_count += 1
    
    print(f"Custom implementation score: {correct_count}/16")
    
    # Print the actual mapping
    print("\nActual mapping from custom implementation:")
    for i in range(16):
        input_binary = format(i, '04b')
        output_binary = format(results[i], '04b')
        expected_binary = format(s_box_table[i], '04b')
        
        if results[i] == s_box_table[i]:
            print(f"✓ {input_binary} -> {output_binary} (Expected: {expected_binary})")
        else:
            print(f"✗ {input_binary} -> {output_binary} (Expected: {expected_binary})")

def create_truth_table_sbox():
    """Create an S-box implementation based directly on the truth table"""
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    # Create an implementation based on multiple controlled X gates
    def truth_table_sbox(circuit, qubits):
        # Add ancilla qubits for the implementation
        anc = QuantumRegister(4, 'anc')
        circuit.add_register(anc)
        
        # Initialize ancilla qubits to 0
        for i in range(4):
            circuit.reset(anc[i])
        
        # For each possible input, add controlled-X gates to the correct output bits
        for input_val in range(16):
            output_val = s_box_table[input_val]
            
            # Convert to binary
            input_binary = format(input_val, '04b')
            output_binary = format(output_val, '04b')
            
            # Create control pattern (X gates before and after for 0 bits)
            for i in range(4):
                if input_binary[i] == '0':
                    circuit.x(qubits[3-i])  # Apply X gate to flip 0->1 for control
            
            # Apply multi-controlled X to the output bits that need to be 1
            for i in range(4):
                if output_binary[i] == '1':
                    # Use the first ancilla as target for the MCX, then CNOT to actual output
                    circuit.mcx(qubits, anc[0])
                    circuit.cx(anc[0], anc[i])
                    # Uncompute the ancilla
                    circuit.mcx(qubits, anc[0])
            
            # Reset control pattern
            for i in range(4):
                if input_binary[i] == '0':
                    circuit.x(qubits[3-i])
        
        # Copy results from ancilla to output qubits
        for i in range(4):
            circuit.swap(qubits[i], anc[i])
        
        return circuit
    
    # Test it
    correct_count = 0
    
    for input_val in range(16):
        expected_output = s_box_table[input_val]
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Set input
        input_binary = format(input_val, '04b')
        for i in range(4):
            if input_binary[3-i] == '1':
                circuit.x(qr[i])
        
        # Apply truth table S-box
        circuit = truth_table_sbox(circuit, qr)
        
        # Measure
        circuit.measure(qr, cr)
        
        # Simulate
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        output = list(counts.keys())[0]
        output_val = int(output, 2)
        
        if output_val == expected_output:
            correct_count += 1
    
    print(f"Truth table implementation score: {correct_count}/16")

if __name__ == "__main__":
    print("Starting advanced S-box correction analysis...")
    
    print("\nTrying alternative S-box implementations:")
    try_alternative_sbox_implementations()
    
    print("\nTesting custom S-box implementation:")
    try_custom_sbox_implementation()
    
    print("\nFinding optimal transformations:")
    best_config = find_optimal_transformation()