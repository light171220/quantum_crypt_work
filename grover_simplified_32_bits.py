from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt

def s_box_8bit(circuit, qubits):
    circuit.cx(qubits[6], qubits[4])
    circuit.cx(qubits[7], qubits[3])
    circuit.cx(qubits[0], qubits[5])
    circuit.cx(qubits[1], qubits[2])
    
    circuit.ccx(qubits[7], qubits[4], qubits[1])
    circuit.ccx(qubits[3], qubits[0], qubits[6])
    
    circuit.ccx(qubits[5], qubits[2], qubits[7])
    circuit.ccx(qubits[1], qubits[6], qubits[4])
    
    circuit.ccx(qubits[2], qubits[0], qubits[3])
    circuit.ccx(qubits[4], qubits[7], qubits[5])
    
    circuit.x(qubits[2])
    circuit.x(qubits[5])
    circuit.cx(qubits[4], qubits[0])
    circuit.cx(qubits[7], qubits[1])
    circuit.cx(qubits[3], qubits[6])
    circuit.cx(qubits[5], qubits[2])
    
    return circuit

def inverse_s_box_8bit(circuit, qubits):
    circuit.cx(qubits[5], qubits[2])
    circuit.cx(qubits[3], qubits[6])
    circuit.cx(qubits[7], qubits[1])
    circuit.cx(qubits[4], qubits[0])
    circuit.x(qubits[5])
    circuit.x(qubits[2])
    
    circuit.ccx(qubits[4], qubits[7], qubits[5])
    circuit.ccx(qubits[2], qubits[0], qubits[3])
    
    circuit.ccx(qubits[1], qubits[6], qubits[4])
    circuit.ccx(qubits[5], qubits[2], qubits[7])
    
    circuit.ccx(qubits[3], qubits[0], qubits[6])
    circuit.ccx(qubits[7], qubits[4], qubits[1])
    
    circuit.cx(qubits[1], qubits[2])
    circuit.cx(qubits[0], qubits[5])
    circuit.cx(qubits[7], qubits[3])
    circuit.cx(qubits[6], qubits[4])
    
    return circuit

def mix_columns_32bit(circuit, qubits):
    circuit.cx(qubits[0], qubits[14])
    circuit.cx(qubits[10], qubits[6])
    circuit.cx(qubits[8], qubits[4])
    circuit.cx(qubits[2], qubits[12])
    circuit.cx(qubits[15], qubits[8])
    circuit.cx(qubits[5], qubits[10])
    circuit.cx(qubits[7], qubits[0])
    circuit.cx(qubits[13], qubits[2])
    
    circuit.cx(qubits[16], qubits[30])
    circuit.cx(qubits[26], qubits[22])
    circuit.cx(qubits[24], qubits[20])
    circuit.cx(qubits[18], qubits[28])
    circuit.cx(qubits[31], qubits[24])
    circuit.cx(qubits[21], qubits[26])
    circuit.cx(qubits[23], qubits[16])
    circuit.cx(qubits[29], qubits[18])
    
    return circuit

def inverse_mix_columns_32bit(circuit, qubits):
    circuit.cx(qubits[29], qubits[18])
    circuit.cx(qubits[23], qubits[16])
    circuit.cx(qubits[21], qubits[26])
    circuit.cx(qubits[31], qubits[24])
    circuit.cx(qubits[18], qubits[28])
    circuit.cx(qubits[24], qubits[20])
    circuit.cx(qubits[26], qubits[22])
    circuit.cx(qubits[16], qubits[30])
    
    circuit.cx(qubits[13], qubits[2])
    circuit.cx(qubits[7], qubits[0])
    circuit.cx(qubits[5], qubits[10])
    circuit.cx(qubits[15], qubits[8])
    circuit.cx(qubits[2], qubits[12])
    circuit.cx(qubits[8], qubits[4])
    circuit.cx(qubits[10], qubits[6])
    circuit.cx(qubits[0], qubits[14])
    
    return circuit

def shift_rows_32bit(circuit, qubits):
    circuit.swap(qubits[8], qubits[12])
    circuit.swap(qubits[9], qubits[13])
    circuit.swap(qubits[10], qubits[14])
    circuit.swap(qubits[11], qubits[15])
    
    circuit.swap(qubits[16], qubits[24])
    circuit.swap(qubits[17], qubits[25])
    circuit.swap(qubits[18], qubits[26])
    circuit.swap(qubits[19], qubits[27])
    
    circuit.swap(qubits[28], qubits[20])
    circuit.swap(qubits[21], qubits[29])
    circuit.swap(qubits[22], qubits[30])
    circuit.swap(qubits[23], qubits[31])
    
    return circuit

def inverse_shift_rows_32bit(circuit, qubits):
    circuit.swap(qubits[23], qubits[31])
    circuit.swap(qubits[22], qubits[30])
    circuit.swap(qubits[21], qubits[29])
    circuit.swap(qubits[28], qubits[20])
    
    circuit.swap(qubits[19], qubits[27])
    circuit.swap(qubits[18], qubits[26])
    circuit.swap(qubits[17], qubits[25])
    circuit.swap(qubits[16], qubits[24])
    
    circuit.swap(qubits[11], qubits[15])
    circuit.swap(qubits[10], qubits[14])
    circuit.swap(qubits[9], qubits[13])
    circuit.swap(qubits[8], qubits[12])
    
    return circuit

def add_round_key(circuit, data_qubits, key_qubits):
    for i in range(32):
        circuit.cx(key_qubits[i], data_qubits[i])
    return circuit

def key_expansion_32bit(circuit, key_qubits, constant):
    W0 = key_qubits[0:16]
    W1 = key_qubits[16:32]
    
    for i in range(16):
        if constant[i] == '1':
            circuit.x(W0[i])
    
    for i in range(8):
        circuit.swap(W1[i], W1[i+8])
    
    circuit = s_box_8bit(circuit, W1[0:8])
    circuit = s_box_8bit(circuit, W1[8:16])
    
    for i in range(16):
        circuit.cx(W1[i], W0[i])
    
    circuit = inverse_s_box_8bit(circuit, W1[0:8])
    circuit = inverse_s_box_8bit(circuit, W1[8:16])
    
    for i in range(8):
        circuit.swap(W1[i], W1[i+8])
    
    for i in range(16):
        circuit.cx(W0[i], W1[i])
    
    return circuit

def inverse_key_expansion_32bit(circuit, key_qubits, constant):
    W0 = key_qubits[0:16]
    W1 = key_qubits[16:32]
    
    for i in range(16):
        circuit.cx(W0[i], W1[i])
    
    for i in range(8):
        circuit.swap(W1[i], W1[i+8])
    
    circuit = s_box_8bit(circuit, W1[0:8])
    circuit = s_box_8bit(circuit, W1[8:16])
    
    for i in range(16):
        circuit.cx(W1[i], W0[i])
    
    circuit = inverse_s_box_8bit(circuit, W1[0:8])
    circuit = inverse_s_box_8bit(circuit, W1[8:16])
    
    for i in range(8):
        circuit.swap(W1[i], W1[i+8])
    
    for i in range(16):
        if constant[i] == '1':
            circuit.x(W0[i])
    
    return circuit

def oracle_circuit_32bit(circuit, data_qubits, key_qubits, target_qubit, known_plaintext, known_ciphertext):
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    for i in range(0, 32, 8):
        circuit = s_box_8bit(circuit, data_qubits[i:i+8])
    circuit.barrier()
    
    circuit = shift_rows_32bit(circuit, data_qubits)
    circuit.barrier()
    
    circuit = mix_columns_32bit(circuit, data_qubits)
    circuit.barrier()
    
    circuit = key_expansion_32bit(circuit, key_qubits, "1000000000000000")
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    for i in range(0, 32, 8):
        circuit = s_box_8bit(circuit, data_qubits[i:i+8])
    circuit.barrier()
    
    circuit = shift_rows_32bit(circuit, data_qubits)
    circuit.barrier()
    
    circuit = key_expansion_32bit(circuit, key_qubits, "0011000000000000")
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    for i in range(32):
        if known_ciphertext[i] == '0':
            circuit.x(data_qubits[i])
    
    circuit.mcx(data_qubits, target_qubit)
    
    for i in range(32):
        if known_ciphertext[i] == '0':
            circuit.x(data_qubits[i])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = inverse_key_expansion_32bit(circuit, key_qubits, "0011000000000000")
    circuit.barrier()
    
    circuit = inverse_shift_rows_32bit(circuit, data_qubits)
    circuit.barrier()
    
    for i in range(24, -1, -8):
        circuit = inverse_s_box_8bit(circuit, data_qubits[i:i+8])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = inverse_key_expansion_32bit(circuit, key_qubits, "1000000000000000")
    circuit.barrier()
    
    circuit = inverse_mix_columns_32bit(circuit, data_qubits)
    circuit.barrier()
    
    circuit = inverse_shift_rows_32bit(circuit, data_qubits)
    circuit.barrier()
    
    for i in range(24, -1, -8):
        circuit = inverse_s_box_8bit(circuit, data_qubits[i:i+8])
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    
    return circuit

def grover_diffusion_32bit(circuit, key_qubits):
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

def run_grover_saes_32bit(plaintext, ciphertext, num_iterations=1):
    key_register = QuantumRegister(32, 'key')
    data_register = QuantumRegister(32, 'data')
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(32, 'classical')
    
    circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)
    
    for i in range(32):
        if plaintext[i] == '1':
            circuit.x(data_register[i])
    
    for qubit in key_register:
        circuit.h(qubit)
    
    circuit.x(target_register)
    circuit.h(target_register)
    
    for _ in range(num_iterations):
        circuit = oracle_circuit_32bit(circuit, data_register, key_register,
                                  target_register[0], plaintext, ciphertext)
        circuit = grover_diffusion_32bit(circuit, key_register)
    
    circuit.measure(key_register, classical_register)
    
    return circuit

plaintext = '01101101001001010110110100100101'
ciphertext = '11001110001000101100111000100010'

def analyze_circuit_resources():
    test_circuit = run_grover_saes_32bit(plaintext, ciphertext, num_iterations=1)
    
    depth = test_circuit.depth()
    gate_counts = test_circuit.count_ops()
    qubit_count = test_circuit.num_qubits
    
    print(f"Circuit depth: {depth}")
    print(f"Total qubits: {qubit_count}")
    print(f"Gate counts: {gate_counts}")
    
    return depth, qubit_count, gate_counts