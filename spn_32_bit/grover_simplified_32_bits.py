from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
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

def bits_to_string(bits):
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        bytes_list.append(int(byte, 2))
    return bytes(bytes_list).decode('ascii', errors='replace')

def string_to_bits(text, bits_length=32):
    byte_array = text.encode('ascii')
    bits = ''.join(format(byte, '08b') for byte in byte_array)
    
    if len(bits) < bits_length:
        bits = bits.zfill(bits_length)
    elif len(bits) > bits_length:
        bits = bits[:bits_length]
        
    return bits

def encrypt_message(message, key, num_rounds=3):
    message_bits = string_to_bits(message, 32)
    key_bits = string_to_bits(key, 32)
    
    circuit = create_encryption_circuit(message_bits, key_bits, num_rounds)
    
    simulator = AerSimulator()
    result = simulator.run(circuit).result()
    counts = result.get_counts()
    
    encrypted_bits = max(counts, key=counts.get)
    
    return encrypted_bits

def decrypt_message(encrypted_bits, key, num_rounds=3):
    key_bits = string_to_bits(key, 32)
    
    circuit = create_decryption_circuit(encrypted_bits, key_bits, num_rounds)
    
    simulator = AerSimulator()
    result = simulator.run(circuit).result()
    counts = result.get_counts()
    
    decrypted_bits = max(counts, key=counts.get)
    
    return bits_to_string(decrypted_bits)

def main():
    key = "KEY!"
    message = "TEST"
    
    print(f"Original message: {message}")
    print(f"Key: {key}")
    
    num_rounds = 3
    
    encrypted_bits = encrypt_message(message, key, num_rounds)
    print(f"Encrypted (bits): {encrypted_bits}")
    
    decrypted_message = decrypt_message(encrypted_bits, key, num_rounds)
    print(f"Decrypted message: {decrypted_message}")
    
    key_bits = string_to_bits(key, 32)
    message_bits = string_to_bits(message, 32)
    
    small_circuit = create_encryption_circuit(message_bits, key_bits, num_rounds=1)
    print("\nExample Encryption Circuit (1 round):")
    print(small_circuit.draw(output='text', fold=80))
    
    test_message = "ABCD"
    print(f"\nTesting with message: {test_message}")
    
    encrypted_bits = encrypt_message(test_message, key, num_rounds)
    decrypted_message = decrypt_message(encrypted_bits, key, num_rounds)
    
    print(f"Encrypted (bits): {encrypted_bits}")
    print(f"Decrypted message: {decrypted_message}")
    
    if decrypted_message == test_message:
        print("Encryption and decryption successful!")
    else:
        print("There was an issue with encryption/decryption.")

if __name__ == "__main__":
    main()