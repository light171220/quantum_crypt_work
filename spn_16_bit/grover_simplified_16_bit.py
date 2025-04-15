from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt


def optimized_s_box_4bit(circuit, qubits):
    circuit.swap(qubits[1], qubits[2])
    circuit.swap(qubits[0], qubits[3])
    
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
    
    circuit.swap(qubits[0], qubits[1])
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[1], qubits[2])
    
    return circuit

def inverse_optimized_s_box_4bit(circuit, qubits):
    circuit.swap(qubits[1], qubits[2])
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[0], qubits[1])
    
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
    
    circuit.swap(qubits[0], qubits[3])
    circuit.swap(qubits[1], qubits[2])
    
    return circuit

def optimized_mix_columns(circuit, qubits):
    circuit.cx(qubits[0], qubits[6])
    circuit.cx(qubits[5], qubits[3])
    circuit.cx(qubits[4], qubits[2])
    circuit.cx(qubits[1], qubits[7])
    circuit.cx(qubits[7], qubits[4])
    circuit.cx(qubits[2], qubits[5])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[6], qubits[1])
    
    circuit.swap(qubits[0], qubits[2])
    circuit.swap(qubits[1], qubits[4])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[4], qubits[6])
    
    return circuit

def inverse_optimized_mix_columns(circuit, qubits):
    circuit.swap(qubits[4], qubits[6])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[1], qubits[4])
    circuit.swap(qubits[0], qubits[2])
    
    circuit.cx(qubits[6], qubits[1])
    circuit.cx(qubits[3], qubits[0])
    circuit.cx(qubits[2], qubits[5])
    circuit.cx(qubits[7], qubits[4])
    circuit.cx(qubits[1], qubits[7])
    circuit.cx(qubits[4], qubits[2])
    circuit.cx(qubits[5], qubits[3])
    circuit.cx(qubits[0], qubits[6])
    
    return circuit

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

def add_round_key(circuit, data_qubits, key_qubits):
    for i in range(16):
        circuit.cx(key_qubits[i], data_qubits[i])
    return circuit

def key_expansion(circuit, key_qubits, constant):
    W0 = key_qubits[0:8]
    W1 = key_qubits[8:16]

    for i in range(8):
        if constant[i] == '1':
            circuit.x(W0[i])

    for i in range(4):
        circuit.swap(W1[i], W1[i+4])

    circuit = optimized_s_box_4bit(circuit, W1[0:4])
    circuit = optimized_s_box_4bit(circuit, W1[4:8])

    for i in range(8):
        circuit.cx(W1[i], W0[i])

    circuit = inverse_optimized_s_box_4bit(circuit, W1[0:4])
    circuit = inverse_optimized_s_box_4bit(circuit, W1[4:8])

    for i in range(4):
        circuit.swap(W1[i], W1[i+4])

    for i in range(8):
        circuit.cx(W0[i], W1[i])

    return circuit

def inverse_key_expansion(circuit, key_qubits, constant):
    W0 = key_qubits[0:8]
    W1 = key_qubits[8:16]

    for i in range(8):
        circuit.cx(W0[i], W1[i])

    for i in range(4):
        circuit.swap(W1[i], W1[i+4])

    circuit = optimized_s_box_4bit(circuit, W1[0:4])
    circuit = optimized_s_box_4bit(circuit, W1[4:8])

    for i in range(8):
        circuit.cx(W1[i], W0[i])

    circuit = inverse_optimized_s_box_4bit(circuit, W1[0:4])
    circuit = inverse_optimized_s_box_4bit(circuit, W1[4:8])

    for i in range(4):
        circuit.swap(W1[i], W1[i+4])

    for i in range(8):
        if constant[i] == '1':
            circuit.x(W0[i])

    return circuit

def oracle_circuit(circuit, data_qubits, key_qubits, target_qubit, known_plaintext, known_ciphertext):
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = optimized_s_box_4bit(circuit, data_qubits[0:4])
    circuit = optimized_s_box_4bit(circuit, data_qubits[4:8])
    circuit = optimized_s_box_4bit(circuit, data_qubits[8:12])
    circuit = optimized_s_box_4bit(circuit, data_qubits[12:16])
    circuit.barrier()
    
    circuit = shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    circuit = optimized_mix_columns(circuit, data_qubits[0:8])
    circuit = optimized_mix_columns(circuit, data_qubits[8:16])
    circuit.barrier()
    
    circuit = key_expansion(circuit, key_qubits, "10000000")
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = optimized_s_box_4bit(circuit, data_qubits[0:4])
    circuit = optimized_s_box_4bit(circuit, data_qubits[4:8])
    circuit = optimized_s_box_4bit(circuit, data_qubits[8:12])
    circuit = optimized_s_box_4bit(circuit, data_qubits[12:16])
    circuit.barrier()
    
    circuit = shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    circuit = key_expansion(circuit, key_qubits, "00110000")
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    for i in range(16):
        if known_ciphertext[i] == '0':
            circuit.x(data_qubits[i])
    circuit.mcx(data_qubits, target_qubit)
    for i in range(16):
        if known_ciphertext[i] == '0':
            circuit.x(data_qubits[i])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = inverse_key_expansion(circuit, key_qubits, "00110000")
    circuit.barrier()
    
    circuit = inverse_shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[12:16])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[8:12])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[4:8])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[0:4])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = inverse_key_expansion(circuit, key_qubits, "10000000")
    circuit.barrier()
    
    circuit = inverse_optimized_mix_columns(circuit, data_qubits[8:16])
    circuit = inverse_optimized_mix_columns(circuit, data_qubits[0:8])
    circuit.barrier()
    
    circuit = inverse_shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[12:16])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[8:12])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[4:8])
    circuit = inverse_optimized_s_box_4bit(circuit, data_qubits[0:4])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)

    return circuit

def grover_diffusion(circuit, key_qubits):
    for qubit in key_qubits:
        circuit.h(qubit)

    for qubit in key_qubits:
        circuit.x(qubit)

    circuit.h(key_qubits[-1])
    circuit.mcx(key_qubits[:-1], key_qubits[-1])
    circuit.h(key_qubits[-1])

    for qubit in key_qubits:
        circuit.x(qubit)

    for qubit in key_qubits:
        circuit.h(qubit)

    return circuit

def encrypt_spn(plaintext, key):
    key_register = QuantumRegister(16, 'key')
    data_register = QuantumRegister(16, 'data')
    classical_register = ClassicalRegister(16, 'classical')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    for i in range(16):
        if plaintext[i] == '1':
            circuit.x(data_register[i])
    
    for i in range(16):
        if key[i] == '1':
            circuit.x(key_register[i])
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit = optimized_s_box_4bit(circuit, data_register[0:4])
    circuit = optimized_s_box_4bit(circuit, data_register[4:8])
    circuit = optimized_s_box_4bit(circuit, data_register[8:12])
    circuit = optimized_s_box_4bit(circuit, data_register[12:16])
    
    circuit = shift_rows(circuit, data_register)
    
    circuit = optimized_mix_columns(circuit, data_register[0:8])
    circuit = optimized_mix_columns(circuit, data_register[8:16])
    
    circuit = key_expansion(circuit, key_register, "10000000")
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit = optimized_s_box_4bit(circuit, data_register[0:4])
    circuit = optimized_s_box_4bit(circuit, data_register[4:8])
    circuit = optimized_s_box_4bit(circuit, data_register[8:12])
    circuit = optimized_s_box_4bit(circuit, data_register[12:16])
    
    circuit = shift_rows(circuit, data_register)
    
    circuit = key_expansion(circuit, key_register, "00110000")
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit.measure(data_register, classical_register)
    
    simulator = AerSimulator(method='matrix_product_state')
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    ciphertext = max(counts, key=counts.get)
    return ciphertext[::-1]

def decrypt_spn(ciphertext, key):
    key_register = QuantumRegister(16, 'key')
    data_register = QuantumRegister(16, 'data')
    classical_register = ClassicalRegister(16, 'classical')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    for i in range(16):
        if ciphertext[i] == '1':
            circuit.x(data_register[i])
    
    for i in range(16):
        if key[i] == '1':
            circuit.x(key_register[i])
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit = inverse_key_expansion(circuit, key_register, "00110000")
    
    circuit = inverse_shift_rows(circuit, data_register)
    
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[12:16])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[8:12])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[4:8])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[0:4])
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit = inverse_key_expansion(circuit, key_register, "10000000")
    
    circuit = inverse_optimized_mix_columns(circuit, data_register[8:16])
    circuit = inverse_optimized_mix_columns(circuit, data_register[0:8])
    
    circuit = inverse_shift_rows(circuit, data_register)
    
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[12:16])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[8:12])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[4:8])
    circuit = inverse_optimized_s_box_4bit(circuit, data_register[0:4])
    
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit.measure(data_register, classical_register)
    
    simulator = AerSimulator(method='matrix_product_state')
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    plaintext = max(counts, key=counts.get)
    return plaintext[::-1]

def run_grover_attack(plaintext, ciphertext, num_iterations=1):
    key_register = QuantumRegister(16, 'key')
    data_register = QuantumRegister(16, 'data')
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(16, 'classical')

    circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)

    for i in range(16):
        if plaintext[i] == '1':
            circuit.x(data_register[i])

    for qubit in key_register:
        circuit.h(qubit)

    circuit.x(target_register)
    circuit.h(target_register)

    for _ in range(num_iterations):
        circuit = oracle_circuit(circuit, data_register, key_register,
                                target_register[0], plaintext, ciphertext)
        circuit = grover_diffusion(circuit, key_register)

    circuit.measure(key_register, classical_register)
    
    simulator = AerSimulator(method='matrix_product_state')
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_counts

def verify_spn():
    plaintext = "1101101100100101"
    key = "0011001000110100"
    
    print("Testing SPN cipher with:")
    print(f"Plaintext: {plaintext}")
    print(f"Key:       {key}")
    
    ciphertext = encrypt_spn(plaintext, key)
    print(f"Ciphertext: {ciphertext}")
    
    recovered_plaintext = decrypt_spn(ciphertext, key)
    print(f"Recovered plaintext: {recovered_plaintext}")
    
    if recovered_plaintext == plaintext:
        print("Encryption/decryption test PASSED!")
    else:
        print("Encryption/decryption test FAILED!")
        
    print("\nRunning Grover attack with limited iterations...")
    top_keys = run_grover_attack(plaintext, ciphertext, num_iterations=1)
    
    print("Top key candidates:")
    for key_candidate, count in top_keys[:5]:
        key_candidate_rev = key_candidate[::-1]
        print(f"Key: {key_candidate_rev}, Count: {count}")
        
    if key in [k[::-1] for k, _ in top_keys[:5]]:
        print("Correct key is among top 5 candidates!")
    else:
        print("Correct key not found in top candidates.")

if __name__ == "__main__":
    verify_spn()