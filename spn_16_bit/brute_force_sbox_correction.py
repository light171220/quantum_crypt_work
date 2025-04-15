from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import itertools

def s_box_with_extended_transformations(circuit, qubits, swaps=None, bit_flips=None, pre_swaps=None, post_swaps=None):
    """
    Extended S-box transformation with more flexible swap configurations
    """
    # Convert None to empty lists to simplify logic
    swaps = swaps or []
    bit_flips = bit_flips or []
    pre_swaps = pre_swaps or []
    post_swaps = post_swaps or []

    # Pre-transformation swaps
    for swap in pre_swaps:
        circuit.swap(qubits[swap[0]], qubits[swap[1]])
    
    # Original S-box implementation
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
    
    # Apply swaps
    for swap in swaps:
        circuit.swap(qubits[swap[0]], qubits[swap[1]])
    
    # Apply bit flips (NOT gates)
    for i in bit_flips:
        circuit.x(qubits[i])
    
    # Post-transformation swaps
    for swap in post_swaps:
        circuit.swap(qubits[swap[0]], qubits[swap[1]])
    
    return circuit

def enhanced_find_optimal_transformation():
    """
    Enhanced version of find_optimal_transformation with more comprehensive search strategies
    """
    # Original S-box mapping to match
    s_box_table = {
        0: 9, 1: 4, 2: 10, 3: 11,
        4: 13, 5: 1, 6: 8, 7: 5,
        8: 6, 9: 2, 10: 0, 11: 3,
        12: 12, 13: 14, 14: 15, 15: 7
    }
    
    def test_transformation(swaps=None, bit_flips=None, input_reordering=None, output_reordering=None, 
                            pre_swaps=None, post_swaps=None):
        """
        Test a specific transformation configuration
        """
        correct_count = 0
        results = {}
        
        for input_val in range(16):
            # Apply input reordering if specified
            if input_reordering:
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
            
            # Apply S-box with extended transformations
            circuit = s_box_with_extended_transformations(
                circuit, qr, 
                swaps=swaps, 
                bit_flips=bit_flips, 
                pre_swaps=pre_swaps, 
                post_swaps=post_swaps
            )
            
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
    
    # Prepare to track the best configuration
    best_score = 0
    best_config = None
    
    # Generate all possible combinations of swaps, bit flips, and reorderings
    all_perms = list(itertools.permutations(range(4)))
    
    # Comprehensive swap configurations
    all_swap_combinations = []
    for num_swaps in range(1, 5):  # 1 to 4 swaps
        for swaps in itertools.combinations(itertools.combinations(range(4), 2), num_swaps):
            all_swap_combinations.append(tuple(swaps))
    
    # Add empty swap configuration
    all_swap_combinations.append(())
    
    # Search through different configurations
    config_attempts = 0
    max_attempts = 10000  # Prevent infinite search
    
    for swaps in all_swap_combinations:
        for num_flips in range(5):  # 0 to 4 bit flips
            for bit_flips in itertools.combinations(range(4), num_flips):
                # Try various input and output reorderings
                for input_perm in [None] + all_perms[:5]:  # Limit to reduce search space
                    for output_perm in [None] + all_perms[:5]:  # Limit to reduce search space
                        # Try pre and post swaps
                        pre_post_swap_combos = [None] + list(itertools.combinations(range(4), 2))[:3]
                        
                        for pre_swaps_config in pre_post_swap_combos:
                            for post_swaps_config in pre_post_swap_combos:
                                # Prevent duplicate attempts
                                config_attempts += 1
                                if config_attempts > max_attempts:
                                    print("Max search attempts reached.")
                                    return best_config
                                
                                # Convert to lists to ensure compatibility
                                pre_swaps = list([pre_swaps_config]) if pre_swaps_config else None
                                post_swaps = list([post_swaps_config]) if post_swaps_config else None
                                
                                # Test the configuration
                                try:
                                    score, mapping = test_transformation(
                                        swaps=swaps, 
                                        bit_flips=bit_flips, 
                                        input_reordering=input_perm, 
                                        output_reordering=output_perm,
                                        pre_swaps=pre_swaps,
                                        post_swaps=post_swaps
                                    )
                                except Exception as e:
                                    print(f"Error in configuration: {e}")
                                    continue
                                
                                # Update best configuration if needed
                                if score > best_score:
                                    best_score = score
                                    best_config = {
                                        "swaps": swaps,
                                        "bit_flips": bit_flips,
                                        "input_reordering": input_perm,
                                        "output_reordering": output_perm,
                                        "pre_swaps": pre_swaps,
                                        "post_swaps": post_swaps,
                                        "score": score,
                                        "mapping": mapping
                                    }
                                    
                                    # Print progress and details of best configuration
                                    print(f"New best found: score {score}/16")
                                    print(f"  Swaps: {swaps}")
                                    print(f"  Bit flips: {bit_flips}")
                                    print(f"  Input reordering: {input_perm}")
                                    print(f"  Output reordering: {output_perm}")
                                    print(f"  Pre-swaps: {pre_swaps}")
                                    print(f"  Post-swaps: {post_swaps}")
                                    print("-" * 50)
                                    
                                    # Early stopping if perfect match found
                                    if score == 16:
                                        print("Perfect match found!")
                                        return best_config
    
    # Print final best configuration
    if best_config:
        print("\nBest configuration found:")
        print(f"  Score: {best_config['score']}/16")
        print(f"  Swaps: {best_config['swaps']}")
        print(f"  Bit flips: {best_config['bit_flips']}")
        print(f"  Input reordering: {best_config['input_reordering']}")
        print(f"  Output reordering: {best_config['output_reordering']}")
        print(f"  Pre-swaps: {best_config['pre_swaps']}")
        print(f"  Post-swaps: {best_config['post_swaps']}")
    else:
        print("No suitable configuration found.")
    
    return best_config

# Run the enhanced search
if __name__ == "__main__":
    print("Starting enhanced S-box transformation search...")
    enhanced_find_optimal_transformation()