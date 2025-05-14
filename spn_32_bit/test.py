from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from collections import Counter
import math

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

def inverse_s_box_4bit(circuit, qubits):
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

    return circuit

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

def add_round_key(circuit, data_qubits, key_qubits):
    for i in range(32):
        circuit.cx(key_qubits[i], data_qubits[i])
    
    return circuit

def key_expansion(circuit, key_qubits, round_num):
    for i in range(round_num + 1):
        circuit.swap(key_qubits[i], key_qubits[31 - i])
    
    for i in range(8):
        circuit = s_box_4bit(circuit, key_qubits[i*4:(i+1)*4])
    
    for i in range(8):
        if ((round_num + 1) & (1 << i)) != 0:
            circuit.x(key_qubits[i])
    
    return circuit

def inverse_key_expansion(circuit, key_qubits, round_num):
    for i in range(8):
        if ((round_num + 1) & (1 << i)) != 0:
            circuit.x(key_qubits[i])
    
    for i in range(7, -1, -1):
        circuit = inverse_s_box_4bit(circuit, key_qubits[i*4:(i+1)*4])
    
    for i in range(round_num, -1, -1):
        circuit.swap(key_qubits[i], key_qubits[31 - i])
    
    return circuit

def encrypt_round(circuit, data_qubits, key_qubits, round_num):
    for i in range(8):
        circuit = s_box_4bit(circuit, data_qubits[i*4:(i+1)*4])
    
    circuit.barrier()
    
    circuit = permutation_32bit(circuit, data_qubits)
    
    circuit.barrier()
    
    circuit = key_expansion(circuit, key_qubits, round_num)
    
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    
    return circuit

def decrypt_round(circuit, data_qubits, key_qubits, round_num):
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    
    circuit.barrier()
    
    circuit = inverse_key_expansion(circuit, key_qubits, round_num)
    
    circuit.barrier()
    
    circuit = inverse_permutation_32bit(circuit, data_qubits)
    
    circuit.barrier()
    
    for i in range(7, -1, -1):
        circuit = inverse_s_box_4bit(circuit, data_qubits[i*4:(i+1)*4])
    
    return circuit

def create_encryption_circuit(plaintext_bits, key_bits, num_rounds=3):
    data_register = QuantumRegister(32, 'data')
    key_register = QuantumRegister(32, 'key')
    classical_register = ClassicalRegister(32, 'c')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    for i in range(32):
        if plaintext_bits[i] == '1':
            circuit.x(data_register[i])
    
    for i in range(32):
        if key_bits[i] == '1':
            circuit.x(key_register[i])
    
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit.barrier()
    
    for r in range(num_rounds):
        circuit = encrypt_round(circuit, data_register, key_register, r)
        circuit.barrier()
    
    circuit.measure(data_register, classical_register)
    
    return circuit

def create_decryption_circuit(ciphertext_bits, key_bits, num_rounds=3):
    data_register = QuantumRegister(32, 'data')
    key_register = QuantumRegister(32, 'key')
    classical_register = ClassicalRegister(32, 'c')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    for i in range(32):
        if ciphertext_bits[i] == '1':
            circuit.x(data_register[i])
    
    for i in range(32):
        if key_bits[i] == '1':
            circuit.x(key_register[i])
    
    circuit.barrier()
    
    for r in range(num_rounds - 1, -1, -1):
        circuit = decrypt_round(circuit, data_register, key_register, r)
        circuit.barrier()
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit.measure(data_register, classical_register)
    
    return circuit

def oracle_function(circuit, data_qubits, key_qubits, target_qubit, plaintext_bits, ciphertext_bits, num_rounds=3):
    # Initialize plaintext
    for i in range(32):
        if plaintext_bits[i] == '1':
            circuit.x(data_qubits[i])
    
    # Initial key mixing
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    # Encryption rounds
    for r in range(num_rounds):
        for i in range(8):
            circuit = s_box_4bit(circuit, data_qubits[i*4:(i+1)*4])
        circuit.barrier()
        
        circuit = permutation_32bit(circuit, data_qubits)
        circuit.barrier()
        
        circuit = key_expansion(circuit, key_qubits, r)
        circuit.barrier()
        
        circuit = add_round_key(circuit, data_qubits, key_qubits)
        circuit.barrier()
    
    # Check if result matches the ciphertext
    for i in range(32):
        if ciphertext_bits[i] == '0':
            circuit.x(data_qubits[i])
    
    # Apply multi-controlled-X to the target qubit
    circuit.mcx(data_qubits, target_qubit)
    
    # Uncompute
    for i in range(32):
        if ciphertext_bits[i] == '0':
            circuit.x(data_qubits[i])
    
    # Reverse encryption
    for r in range(num_rounds-1, -1, -1):
        circuit = add_round_key(circuit, data_qubits, key_qubits)
        circuit.barrier()
        
        circuit = inverse_key_expansion(circuit, key_qubits, r)
        circuit.barrier()
        
        circuit = inverse_permutation_32bit(circuit, data_qubits)
        circuit.barrier()
        
        for i in range(7, -1, -1):
            circuit = inverse_s_box_4bit(circuit, data_qubits[i*4:(i+1)*4])
        circuit.barrier()
    
    # Reverse initial key mixing
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    
    # Uncompute plaintext
    for i in range(32):
        if plaintext_bits[i] == '1':
            circuit.x(data_qubits[i])
    
    return circuit

def diffuser(circuit, key_qubits):
    # Apply H gates to all qubits
    for qubit in key_qubits:
        circuit.h(qubit)
    
    # Apply X gates to all qubits
    for qubit in key_qubits:
        circuit.x(qubit)
    
    # Apply multi-controlled Z gate
    circuit.h(key_qubits[-1])
    circuit.mcx(key_qubits[:-1], key_qubits[-1])
    circuit.h(key_qubits[-1])
    
    # Apply X gates to all qubits
    for qubit in key_qubits:
        circuit.x(qubit)
    
    # Apply H gates to all qubits
    for qubit in key_qubits:
        circuit.h(qubit)
    
    return circuit

def create_grover_circuit(plaintext_bits, ciphertext_bits, iterations, num_rounds=3):
    data_register = QuantumRegister(32, 'data')
    key_register = QuantumRegister(32, 'key') 
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(32, 'c')
    
    circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)
    
    # Initialize key qubits in superposition
    for qubit in key_register:
        circuit.h(qubit)
    
    # Initialize target qubit
    circuit.x(target_register)
    circuit.h(target_register)
    
    # Apply Grover's algorithm for the specified number of iterations
    for _ in range(iterations):
        circuit = oracle_function(circuit, data_register, key_register, target_register[0], plaintext_bits, ciphertext_bits, num_rounds)
        circuit.barrier()
        circuit = diffuser(circuit, key_register)
        circuit.barrier()
    
    # Measure key qubits
    circuit.measure(key_register, classical_register)
    
    return circuit

def count_gates(circuit):
    gate_counts = Counter()
    for instruction, qargs, cargs in circuit.data:
        gate_name = instruction.name
        gate_counts[gate_name] += 1
    return gate_counts

def get_circuit_statistics(circuit_type, num_rounds=3, grover_iterations=None):
    plaintext_bits = '0' * 32
    key_bits = '0' * 32
    ciphertext_bits = '1' * 32
    
    if circuit_type == "encryption":
        circuit = create_encryption_circuit(plaintext_bits, key_bits, num_rounds)
    elif circuit_type == "decryption":
        circuit = create_decryption_circuit(ciphertext_bits, key_bits, num_rounds)
    elif circuit_type == "grover":
        circuit = create_grover_circuit(plaintext_bits, ciphertext_bits, grover_iterations, num_rounds)
    else:
        return None
    
    depth = circuit.depth()
    gate_counts = count_gates(circuit)
    total_gates = sum(gate_counts.values())
    
    return {
        "circuit_type": f"{circuit_type.capitalize()} ({num_rounds} rounds)" + (f", {grover_iterations} Grover iterations" if grover_iterations else ""),
        "depth": depth,
        "gate_counts": gate_counts,
        "total_gates": total_gates,
        "qubits": circuit.num_qubits
    }

def print_circuit_statistics(stats):
    print(f"\n===== {stats['circuit_type']} CIRCUIT STATISTICS =====")
    print(f"Circuit depth: {stats['depth']}")
    print(f"Total gates: {stats['total_gates']}")
    print(f"Number of qubits: {stats['qubits']}")
    print("\nGate counts:")
    
    sorted_gates = sorted(stats['gate_counts'].items(), key=lambda x: x[1], reverse=True)
    for gate_name, count in sorted_gates:
        print(f"  {gate_name}: {count}")

def analyze_grover_scaling(max_iterations=5, num_rounds=3):
    print("\n===== GROVER'S ALGORITHM SCALING ANALYSIS =====")
    
    # Theoretical number of iterations for 32-bit key
    n = 32  # number of key bits
    optimal_iterations = round(math.pi/4 * math.sqrt(2**n))
    
    print(f"For a {n}-bit key search:")
    print(f"Theoretical optimal number of Grover iterations: {optimal_iterations}")
    print(f"Search space size: 2^{n} = {2**n}")
    print(f"Classical search (average): {2**(n-1)} operations")
    print(f"Quantum search (Grover): ~{round(math.sqrt(2**n))} operations\n")
    
    results = []
    
    for i in range(1, max_iterations + 1):
        stats = get_circuit_statistics("grover", num_rounds, i)
        results.append({
            "iterations": i,
            "depth": stats["depth"],
            "total_gates": stats["total_gates"]
        })
        
        print(f"Grover iterations: {i}")
        print(f"  Circuit depth: {stats['depth']}")
        print(f"  Total gates: {stats['total_gates']}")
        
        # Estimate success probability
        success_prob = math.sin((2*i + 1) * math.asin(1/math.sqrt(2**n)))**2
        print(f"  Estimated success probability: {success_prob:.10f}")
        
        # Calculate time factor compared to a single iteration
        if i > 1:
            depth_factor = stats["depth"] / results[0]["depth"]
            gates_factor = stats["total_gates"] / results[0]["total_gates"]
            print(f"  Depth factor (compared to 1 iteration): {depth_factor:.2f}x")
            print(f"  Gates factor (compared to 1 iteration): {gates_factor:.2f}x")
        
        print()
    
    # Calculate theoretical practical limit
    practical_limit = int(math.ceil(math.sqrt(2**n) / 10))
    print(f"Practical limit for near-term quantum computers: ~{practical_limit} Grover iterations")
    print(f"This would require approximately {practical_limit * results[0]['depth']} circuit depth")
    print(f"and {practical_limit * results[0]['total_gates']} total gates.")
    print(f"Success probability with {practical_limit} iterations: {math.sin((2*practical_limit + 1) * math.asin(1/math.sqrt(2**n)))**2:.10f}")

if __name__ == "__main__":
    # Analyze basic encryption and decryption for 1, 2, and 3 rounds
    for num_rounds in [1, 2, 3]:
        encrypt_stats = get_circuit_statistics("encryption", num_rounds)
        print_circuit_statistics(encrypt_stats)
        
        decrypt_stats = get_circuit_statistics("decryption", num_rounds)
        print_circuit_statistics(decrypt_stats)
    
    # Analyze Grover's algorithm scaling
    analyze_grover_scaling(max_iterations=3, num_rounds=1)