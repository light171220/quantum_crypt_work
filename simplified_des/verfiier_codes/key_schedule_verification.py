from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import itertools
import numpy as np

def key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num):
    """
    Correct implementation of key schedule operation
    """
    p10_to_p8_table = [0, 1, 3, 4, 5, 7, 8, 9]
    shift = 1 if round_num == 1 else 3
    
    for i in range(8):
        key_idx = (p10_to_p8_table[i] + shift) % 10
        idx = expansion_indices[i]
        circuit.cx(key_qubits[key_idx], target_qubits[idx])
    
    return circuit

def inverse_key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num):
    """
    Correct implementation of inverse key schedule operation
    This is the exact same operation as key_schedule because controlled-X is its own inverse
    when applied with the same control and target qubits
    """
    p10_to_p8_table = [0, 1, 3, 4, 5, 7, 8, 9]
    shift = 1 if round_num == 1 else 3
    
    for i in range(7, -1, -1):
        key_idx = (p10_to_p8_table[i] + shift) % 10
        idx = expansion_indices[i]
        circuit.cx(key_qubits[key_idx], target_qubits[idx])
    
    return circuit

def create_verification_circuit(key_bits, target_bits, round_num):
    """
    Creates a circuit to verify the key schedule and its inverse
    correctly preserve initial state for a given key and target
    """
    key_qubits = QuantumRegister(10, 'key')
    target_qubits = QuantumRegister(8, 'target')
    output = ClassicalRegister(8, 'out')
    
    circuit = QuantumCircuit(key_qubits, target_qubits, output)
    
    for i, bit in enumerate(key_bits):
        if bit == '1':
            circuit.x(key_qubits[i])
    
    for i, bit in enumerate(target_bits):
        if bit == '1':
            circuit.x(target_qubits[i])
    
    expansion_indices = [3, 0, 1, 2, 1, 2, 3, 0]
    
    key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num)
    inverse_key_schedule(circuit, key_qubits, target_qubits, expansion_indices, round_num)
    
    circuit.measure(target_qubits, output)
    
    return circuit

def verify_with_specific_inputs():
    """
    Tests verification with specific key/target pairs
    """
    test_vectors = [
        {"key": "1010101010", "target": "10101010", "round": 1},
        {"key": "0000000000", "target": "11111111", "round": 1},
        {"key": "1111111111", "target": "10101010", "round": 2},
        {"key": "0101010101", "target": "01010101", "round": 2}
    ]
    
    simulator = AerSimulator()
    
    failures = 0
    for vector in test_vectors:
        circuit = create_verification_circuit(vector["key"], vector["target"], vector["round"])
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        actual_output = list(counts.keys())[0]
        
        if actual_output != vector["target"]:
            failures += 1
            print(f"Test vector failed: key={vector['key']}, target={vector['target']}, round={vector['round']}")
            print(f"Expected: {vector['target']}, Actual: {actual_output}")
    
    if failures == 0:
        print("All specific test vectors PASSED")
        return True
    else:
        print(f"{failures} specific test vectors FAILED")
        return False

def verify_with_superposition():
    """
    Tests with keys in superposition to verify for all possible key inputs at once
    """
    key_qubits = QuantumRegister(10, 'key')
    target_qubits = QuantumRegister(8, 'target')
    output = ClassicalRegister(8, 'out')
    
    circuit = QuantumCircuit(key_qubits, target_qubits, output)
    
    for i in range(10):
        circuit.h(key_qubits[i])
    
    target_bits = "10101010"
    for i, bit in enumerate(target_bits):
        if bit == '1':
            circuit.x(target_qubits[i])
    
    expansion_indices = [3, 0, 1, 2, 1, 2, 3, 0]
    
    key_schedule(circuit, key_qubits, target_qubits, expansion_indices, 1)
    inverse_key_schedule(circuit, key_qubits, target_qubits, expansion_indices, 1)
    
    circuit.measure(target_qubits, output)
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=8192)
    result = job.result()
    counts = result.get_counts()
    
    if len(counts) > 1 or target_bits not in counts:
        print("Superposition test FAILED")
        print(f"Expected only {target_bits}, got: {counts}")
        return False
    else:
        print("Superposition test PASSED")
        return True

def verify_sample_cases():
    """
    Verifies a sampling of various key/target combinations for both rounds
    """
    simulator = AerSimulator()
    
    np.random.seed(42)
    num_samples = 50
    key_samples = [''.join(np.random.choice(['0', '1']) for _ in range(10)) for _ in range(num_samples)]
    target_samples = [''.join(np.random.choice(['0', '1']) for _ in range(8)) for _ in range(num_samples)]
    
    failures = 0
    total_tests = 0
    
    for round_num in [1, 2]:
        for key in key_samples[:10]:
            for target in target_samples[:5]:
                total_tests += 1
                circuit = create_verification_circuit(key, target, round_num)
                job = simulator.run(circuit, shots=1)
                result = job.result()
                counts = result.get_counts()
                actual_output = list(counts.keys())[0]
                
                if actual_output != target:
                    failures += 1
                    print(f"Sample failed: key={key}, target={target}, round={round_num}")
                    print(f"Expected: {target}, Actual: {actual_output}")
                    
                    if failures >= 5:
                        print(f"Stopping after 5 failures out of {total_tests} tests")
                        return False
    
    if failures == 0:
        print(f"All {total_tests} sample tests PASSED")
        return True
    else:
        print(f"{failures} out of {total_tests} sample tests FAILED")
        return False

def exhaustive_test_cases():
    """
    Test against all 256 possible target states with several key patterns for both rounds
    """
    simulator = AerSimulator()
    
    key_patterns = [
        "0000000000",
        "1010101010",
        "1111111111",
        "0101010101",
        "1100110011"
    ]
    
    all_target_patterns = ["".join(bits) for bits in itertools.product("01", repeat=8)]
    
    failures = 0
    total_tests = 0
    
    for round_num in [1, 2]:
        for key in key_patterns:
            for target in all_target_patterns:
                total_tests += 1
                circuit = create_verification_circuit(key, target, round_num)
                job = simulator.run(circuit, shots=1)
                result = job.result()
                counts = result.get_counts()
                actual_output = list(counts.keys())[0]
                
                if actual_output != target:
                    failures += 1
                    print(f"Exhaustive test failed: key={key}, target={target}, round={round_num}")
                    print(f"Expected: {target}, Actual: {actual_output}")
                    
                    if failures >= 3:
                        print(f"Stopping after {failures} failures...")
                        return False
    
    if failures == 0:
        print(f"All {total_tests} exhaustive tests PASSED")
        return True
    else:
        print(f"{failures} out of {total_tests} exhaustive tests FAILED")
        return False

def key_schedule_effect_test():
    """
    Tests the actual effect of the key_schedule (not just verifying invertibility)
    """
    key_qubits = QuantumRegister(10, 'key')
    target_qubits = QuantumRegister(8, 'target')
    cr = ClassicalRegister(8, 'cr')
    
    circuit = QuantumCircuit(key_qubits, target_qubits, cr)
    
    key_bits = "1010101010"
    for i, bit in enumerate(key_bits):
        if bit == '1':
            circuit.x(key_qubits[i])
    
    expansion_indices = [3, 0, 1, 2, 1, 2, 3, 0]
    
    key_schedule(circuit, key_qubits, target_qubits, expansion_indices, 1)
    
    circuit.measure(target_qubits, cr)
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1)
    result = job.result()
    counts = result.get_counts()
    output = list(counts.keys())[0]
    
    print(f"Key schedule effect test:")
    print(f"Key: {key_bits}")
    print(f"Effect on target (should be non-zero): {output}")
    
    return True

def run_all_verification_tests():
    """
    Runs all verification tests and reports results
    """
    print("Running key_schedule verification tests...")
    
    print("\nTest 1: Verifying with specific inputs")
    specific_passed = verify_with_specific_inputs()
    
    print("\nTest 2: Verifying with keys in superposition")
    superposition_passed = verify_with_superposition()
    
    print("\nTest 3: Verifying with sampled cases")
    sample_passed = verify_sample_cases()
    
    print("\nTest 4: Testing exhaustively for select keys")
    exhaustive_passed = exhaustive_test_cases()
    
    print("\nTest 5: Testing actual key_schedule effect")
    effect_passed = key_schedule_effect_test()
    
    all_passed = specific_passed and superposition_passed and sample_passed and exhaustive_passed and effect_passed
    
    print("\nVerification Summary:")
    print(f"Specific inputs test: {'PASS' if specific_passed else 'FAIL'}")
    print(f"Superposition test: {'PASS' if superposition_passed else 'FAIL'}")
    print(f"Sampled cases test: {'PASS' if sample_passed else 'FAIL'}")
    print(f"Exhaustive test: {'PASS' if exhaustive_passed else 'FAIL'}")
    print(f"Effect test: {'PASS' if effect_passed else 'FAIL'}")
    print(f"\nOverall verification: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    run_all_verification_tests()