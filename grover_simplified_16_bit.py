from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import numpy as np
import matplotlib.pyplot as plt

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
    circuit.ccx(qubits[2],qubits[1],qubits[0])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.ccx(qubits[0], qubits[1], qubits[3])

    return circuit

def inverse_s_box_4bit(circuit, qubits):

    circuit.ccx(qubits[0], qubits[1], qubits[3])
    circuit.ccx(qubits[0], qubits[3], qubits[1])
    circuit.ccx(qubits[2],qubits[1],qubits[0])
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

def inverse_mix_columns(circuit, qubits):

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

    circuit = s_box_4bit(circuit, W1[0:4])
    circuit = s_box_4bit(circuit, W1[4:8])

    for i in range(8):
        circuit.cx(W1[i], W0[i])

    circuit = inverse_s_box_4bit(circuit, W1[0:4])
    circuit = inverse_s_box_4bit(circuit, W1[4:8])

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

    circuit = s_box_4bit(circuit, W1[0:4])
    circuit = s_box_4bit(circuit, W1[4:8])

    for i in range(8):
        circuit.cx(W1[i], W0[i])

    circuit = inverse_s_box_4bit(circuit, W1[0:4])
    circuit = inverse_s_box_4bit(circuit, W1[4:8])

    for i in range(4):
        circuit.swap(W1[i], W1[i+4])

    for i in range(8):
        if constant[i] == '1':
            circuit.x(W0[i])

    return circuit

def oracle_circuit(circuit, data_qubits, key_qubits, target_qubit, known_plaintext, known_ciphertext):

    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    circuit = s_box_4bit(circuit, data_qubits[0:4])
    circuit = s_box_4bit(circuit, data_qubits[4:8])
    circuit = s_box_4bit(circuit, data_qubits[8:12])
    circuit = s_box_4bit(circuit, data_qubits[12:16])
    circuit.barrier()
    circuit = shift_rows(circuit, data_qubits)
    circuit.barrier()
    circuit = mix_columns(circuit, data_qubits[0:8])
    circuit = mix_columns(circuit, data_qubits[8:16])
    circuit.barrier()
    circuit = key_expansion(circuit, key_qubits, "10000000")
    circuit.barrier()
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    circuit = s_box_4bit(circuit, data_qubits[0:4])
    circuit = s_box_4bit(circuit, data_qubits[4:8])
    circuit = s_box_4bit(circuit, data_qubits[8:12])
    circuit = s_box_4bit(circuit, data_qubits[12:16])
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
    circuit = inverse_s_box_4bit(circuit, data_qubits[12:16])
    circuit = inverse_s_box_4bit(circuit, data_qubits[8:12])
    circuit = inverse_s_box_4bit(circuit, data_qubits[4:8])
    circuit = inverse_s_box_4bit(circuit, data_qubits[0:4])
    circuit.barrier()
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    circuit = inverse_key_expansion(circuit, key_qubits, "10000000")
    circuit.barrier()
    circuit = inverse_mix_columns(circuit, data_qubits[8:16])
    circuit = inverse_mix_columns(circuit, data_qubits[0:8])
    circuit.barrier()
    circuit = inverse_shift_rows(circuit, data_qubits)
    circuit.barrier()
    circuit = inverse_s_box_4bit(circuit, data_qubits[12:16])
    circuit = inverse_s_box_4bit(circuit, data_qubits[8:12])
    circuit = inverse_s_box_4bit(circuit, data_qubits[4:8])
    circuit = inverse_s_box_4bit(circuit, data_qubits[0:4])
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

key_register = QuantumRegister(16, 'key')
data_register = QuantumRegister(16, 'data')
target_register = QuantumRegister(1, 'target')
classical_register = ClassicalRegister(16, 'classical')

circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)

plaintext = '1101101100100101'
ciphertext = '1100111000100010'

for i in range(16):
    if plaintext[i] == '1':
        circuit.x(data_register[i])

for qubit in key_register:
    circuit.h(qubit)

circuit.x(target_register)
circuit.h(target_register)

# num_iterations = int(np.pi/4 * np.sqrt(2**16))
num_iterations = 1

for _ in range(num_iterations):
    circuit = oracle_circuit(circuit, data_register, key_register,
                            target_register[0], plaintext, ciphertext)
    circuit = grover_diffusion(circuit, key_register)

circuit.measure(key_register, classical_register)