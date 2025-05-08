from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCXGate
import matplotlib.pyplot as plt

def s0_box_circuit(circuit, qubits):
    [q0, q1, q2, q3] = qubits
    
    circuit.ccx(q0, q2, q3)
    circuit.ccx(q0, q1, q3)
    circuit.swap(q1, q3)
    circuit.x(q2)
    circuit.ccx(q0, q1, q2)
    circuit.cx(q3, q1)
    circuit.cx(q2, q0)
    
    return circuit

def s1_box_circuit(circuit, qubits):
    [q0, q1, q2, q3] = qubits
    
    circuit.swap(q3, q1)
    circuit.cx(q3, q2)
    circuit.ccx(q1, q2, q3)
    circuit.swap(q0, q1)
    circuit.ccx(q1, q2, q3)
    circuit.ccx(q0, q1, q3)
    circuit.cx(q1, q3)
    circuit.ccx(q0, q2, q3)
    circuit.swap(q1, q3)
    circuit.swap(q1, q0)
    circuit.ccx(q0, q1, q2)
    circuit.cx(q0, q1)
    circuit.cx(q2, q1)
    
    return circuit

def inverse_s0_box_circuit(circuit, qubits):
    [q0, q1, q2, q3] = qubits
    
    circuit.cx(q2, q0)
    circuit.cx(q3, q1)
    circuit.ccx(q0, q1, q2)
    circuit.x(q2)
    circuit.swap(q1, q3)
    circuit.ccx(q0, q1, q3)
    circuit.ccx(q0, q2, q3)
    
    return circuit

def inverse_s1_box_circuit(circuit, qubits):
    [q0, q1, q2, q3] = qubits
    
    circuit.cx(q2, q1)
    circuit.cx(q0, q1)
    circuit.ccx(q0, q1, q2)
    circuit.swap(q1, q0)
    circuit.swap(q1, q3)
    circuit.ccx(q0, q2, q3)
    circuit.cx(q1, q3)
    circuit.ccx(q0, q1, q3)
    circuit.ccx(q1, q2, q3)
    circuit.swap(q0, q1)
    circuit.cx(q3, q2)
    circuit.swap(q3, q1)
    
    return circuit

def initial_permutation(circuit, qubits):
    circuit.swap(qubits[0], qubits[7])
    circuit.swap(qubits[1], qubits[3])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[4], qubits[6])
    
    return circuit

def final_permutation(circuit, qubits):
    circuit.swap(qubits[4], qubits[6])
    circuit.swap(qubits[2], qubits[5])
    circuit.swap(qubits[1], qubits[3])
    circuit.swap(qubits[0], qubits[7])
    
    return circuit

def swap_halves(circuit, qubits):
    for i in range(4):
        circuit.swap(qubits[i], qubits[i+4])
    return circuit

def direct_key_application(circuit, key_qubits, plaintext_qubits, round_num):
    left_half = plaintext_qubits[0:4]
    right_half = plaintext_qubits[4:8]
    
    if round_num == 1:
        key_to_right_half = {
            0: {5: 3, 0: 0, 6: 1, 1: 2},
            1: {5: 1, 0: 2, 6: 3, 1: 0},
            2: {9: 3, 3: 0, 8: 1, 7: 2},
            3: {9: 1, 3: 2, 8: 3, 7: 0}
        }
    else:
        key_to_right_half = {
            0: {0: 3, 6: 0, 1: 1, 9: 2},
            1: {0: 1, 6: 2, 1: 3, 9: 0},
            2: {3: 3, 8: 0, 7: 1, 2: 2},
            3: {3: 1, 8: 2, 7: 3, 2: 0}
        }
    
    for left_idx, key_map in key_to_right_half.items():
        for key_idx, right_idx in key_map.items():
            circuit.cx(key_qubits[key_idx], plaintext_qubits[4+right_idx])
            circuit.cx(plaintext_qubits[4+right_idx], left_half[left_idx])
    
    return circuit

def encrypt_sdes(circuit, plaintext_qubits, key_qubits):
    circuit = initial_permutation(circuit, plaintext_qubits)
    circuit.barrier()
    
    circuit = direct_key_application(circuit, key_qubits, plaintext_qubits, 1)
    circuit.barrier()
    
    circuit = swap_halves(circuit, plaintext_qubits)
    circuit.barrier()
    
    circuit = direct_key_application(circuit, key_qubits, plaintext_qubits, 2)
    circuit.barrier()
    
    circuit = final_permutation(circuit, plaintext_qubits)
    circuit.barrier()
    
    return circuit

def inverse_direct_key_application(circuit, key_qubits, plaintext_qubits, round_num):
    left_half = plaintext_qubits[0:4]
    right_half = plaintext_qubits[4:8]
    
    if round_num == 1:
        key_to_right_half = {
            0: {5: 3, 0: 0, 6: 1, 1: 2},
            1: {5: 1, 0: 2, 6: 3, 1: 0},
            2: {9: 3, 3: 0, 8: 1, 7: 2},
            3: {9: 1, 3: 2, 8: 3, 7: 0}
        }
    else:
        key_to_right_half = {
            0: {0: 3, 6: 0, 1: 1, 9: 2},
            1: {0: 1, 6: 2, 1: 3, 9: 0},
            2: {3: 3, 8: 0, 7: 1, 2: 2},
            3: {3: 1, 8: 2, 7: 3, 2: 0}
        }
    
    for left_idx, key_map in reversed(list(key_to_right_half.items())):
        for key_idx, right_idx in reversed(list(key_map.items())):
            circuit.cx(plaintext_qubits[4+right_idx], left_half[left_idx])
            circuit.cx(key_qubits[key_idx], plaintext_qubits[4+right_idx])
    
    return circuit

def inverse_sdes(circuit, plaintext_qubits, key_qubits):
    circuit = final_permutation(circuit, plaintext_qubits)
    circuit.barrier()
    
    circuit = inverse_direct_key_application(circuit, key_qubits, plaintext_qubits, 2)
    circuit.barrier()
    
    circuit = swap_halves(circuit, plaintext_qubits)
    circuit.barrier()
    
    circuit = inverse_direct_key_application(circuit, key_qubits, plaintext_qubits, 1)
    circuit.barrier()
    
    circuit = initial_permutation(circuit, plaintext_qubits)
    circuit.barrier()
    
    return circuit

def oracle_function(circuit, plaintext_qubits, key_qubits, target_qubit, ciphertext):
    circuit = encrypt_sdes(circuit, plaintext_qubits, key_qubits)
    
    for i in range(8):
        if ciphertext[i] == '0':
            circuit.x(plaintext_qubits[i])
    
    gate = MCXGate(8)
    qubits_list = [plaintext_qubits[i] for i in range(8)]
    qubits_list.append(target_qubit)
    circuit.append(gate, qubits_list)
    
    for i in range(8):
        if ciphertext[i] == '0':
            circuit.x(plaintext_qubits[i])
    
    circuit = inverse_sdes(circuit, plaintext_qubits, key_qubits)
    
    return circuit

def diffusion(circuit, key_qubits):
    for qubit in key_qubits:
        circuit.h(qubit)
    
    for qubit in key_qubits:
        circuit.x(qubit)
    
    circuit.h(key_qubits[-1])
    
    gate = MCXGate(9)
    control_qubits = [key_qubits[i] for i in range(9)]
    target = key_qubits[9]
    circuit.append(gate, control_qubits + [target])
    
    circuit.h(key_qubits[-1])
    
    for qubit in key_qubits:
        circuit.x(qubit)
    
    for qubit in key_qubits:
        circuit.h(qubit)
    
    return circuit

def grover_sdes(plaintext_value, ciphertext_value, iterations=2):
    plaintext_binary = format(plaintext_value, '08b')
    ciphertext_binary = format(ciphertext_value, '08b')
    
    plaintext = QuantumRegister(8, 'pt')
    key = QuantumRegister(10, 'key')
    target = QuantumRegister(1, 'target')
    cr = ClassicalRegister(10, 'cr')
    circuit = QuantumCircuit(plaintext, key, target, cr)
    
    for i in range(8):
        if plaintext_binary[i] == '1':
            circuit.x(plaintext[i])
    
    for qubit in key:
        circuit.h(qubit)
    
    circuit.x(target)
    circuit.h(target)
    
    for _ in range(iterations):
        circuit = oracle_function(circuit, plaintext, key, target[0], ciphertext_binary)
        circuit = diffusion(circuit, key)
    
    circuit.measure(key, cr)
    
    return circuit

def run_grover(plaintext_val, ciphertext_val, iterations=2, shots=1024):
    plaintext_binary = format(plaintext_val, '08b')
    ciphertext_binary = format(ciphertext_val, '08b')
    
    circuit = grover_sdes(plaintext_val, ciphertext_val, iterations)
    print(circuit.draw())
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    plot_histogram(sorted_counts, title=f'Grover Results: Iterations={iterations}, Plaintext={plaintext_binary}, Ciphertext={ciphertext_binary}')
    plt.tight_layout()
    plt.savefig(f'grover_results_iter{iterations}.png')
    plt.show()
    
    top_key = max(counts, key=counts.get)
    
    print(f"Plaintext: {plaintext_binary}")
    print(f"Ciphertext: {ciphertext_binary}")
    print(f"Most likely key: {top_key} with {counts[top_key]} counts ({counts[top_key]/shots*100:.2f}%)")
    
    return counts

def test_sdes_grover():
    plaintext_val = 0b01010100
    ciphertext_val = 0b11100111
    
    print("Running Grover's algorithm with 1 iteration:")
    counts1 = run_grover(plaintext_val, ciphertext_val, iterations=1, shots=1024)

if __name__ == "__main__":
    test_sdes_grover()