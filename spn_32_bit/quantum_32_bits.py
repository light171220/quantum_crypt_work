from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
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

def sub_bytes(circuit, qubits):
    for i in range(8):
        circuit = optimized_s_box_4bit(circuit, qubits[i*4:(i+1)*4])
    return circuit

def inverse_sub_bytes(circuit, qubits):
    for i in range(7, -1, -1):
        circuit = inverse_optimized_s_box_4bit(circuit, qubits[i*4:(i+1)*4])
    return circuit

def shift_rows(circuit, qubits):
    # Organize 32 bits as 4 rows of 8 bits (similar to AES organization)
    # Row 0: No shift (qubits 0-7)
    # Row 1: Shift left by 2 bits (qubits 8-15)
    circuit.swap(qubits[8], qubits[10])
    circuit.swap(qubits[9], qubits[11])
    circuit.swap(qubits[12], qubits[14])
    circuit.swap(qubits[13], qubits[15])
    
    # Row 2: Shift left by 4 bits (qubits 16-23)
    circuit.swap(qubits[16], qubits[20])
    circuit.swap(qubits[17], qubits[21])
    circuit.swap(qubits[18], qubits[22])
    circuit.swap(qubits[19], qubits[23])
    
    # Row 3: Shift left by 6 bits (qubits 24-31)
    circuit.swap(qubits[24], qubits[30])
    circuit.swap(qubits[25], qubits[31])
    circuit.swap(qubits[26], qubits[28])
    circuit.swap(qubits[27], qubits[29])
    
    return circuit

def inverse_shift_rows(circuit, qubits):
    # Reverse of shift_rows
    # Row 3: Shift right by 6 bits (qubits 24-31)
    circuit.swap(qubits[27], qubits[29])
    circuit.swap(qubits[26], qubits[28])
    circuit.swap(qubits[25], qubits[31])
    circuit.swap(qubits[24], qubits[30])
    
    # Row 2: Shift right by 4 bits (qubits 16-23)
    circuit.swap(qubits[19], qubits[23])
    circuit.swap(qubits[18], qubits[22])
    circuit.swap(qubits[17], qubits[21])
    circuit.swap(qubits[16], qubits[20])
    
    # Row 1: Shift right by 2 bits (qubits 8-15)
    circuit.swap(qubits[13], qubits[15])
    circuit.swap(qubits[12], qubits[14])
    circuit.swap(qubits[9], qubits[11])
    circuit.swap(qubits[8], qubits[10])
    
    return circuit

def mix_columns(circuit, qubits):
    # Mix 4 columns, each column has 8 bits (2 from each row)
    for col in range(4):
        col_start = col * 2
        
        # Apply mix columns to each column
        # These operations simulate the matrix multiplication in AES MixColumns
        # Column 0: bits 0-1, 8-9, 16-17, 24-25
        # Column 1: bits 2-3, 10-11, 18-19, 26-27
        # Column 2: bits 4-5, 12-13, 20-21, 28-29
        # Column 3: bits 6-7, 14-15, 22-23, 30-31
        
        # Row operations (similar to AES Mix Columns multiplication)
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

def add_round_key(circuit, data_qubits, key_qubits):
    for i in range(32):
        circuit.cx(key_qubits[i], data_qubits[i])
    return circuit

def key_expansion(circuit, key_qubits, round_num):
    # Apply round constant
    for i in range(8):
        if ((round_num + 1) & (1 << i)) != 0:
            circuit.x(key_qubits[i])
    
    # Rotate word
    for i in range(8):
        circuit.swap(key_qubits[i], key_qubits[24+i])
    
    # Apply S-box to each 4-bit chunk
    for i in range(8):
        circuit = optimized_s_box_4bit(circuit, key_qubits[i*4:(i+1)*4])
    
    # Mix with previous key parts
    for i in range(8):
        circuit.cx(key_qubits[i], key_qubits[i+8])
        circuit.cx(key_qubits[i+8], key_qubits[i+16])
        circuit.cx(key_qubits[i+16], key_qubits[i+24])
    
    return circuit

def inverse_key_expansion(circuit, key_qubits, round_num):
    # Undo mixing
    for i in range(7, -1, -1):
        circuit.cx(key_qubits[i+16], key_qubits[i+24])
        circuit.cx(key_qubits[i+8], key_qubits[i+16])
        circuit.cx(key_qubits[i], key_qubits[i+8])
    
    # Undo S-box
    for i in range(7, -1, -1):
        circuit = inverse_optimized_s_box_4bit(circuit, key_qubits[i*4:(i+1)*4])
    
    # Undo rotation
    for i in range(7, -1, -1):
        circuit.swap(key_qubits[i], key_qubits[24+i])
    
    # Undo round constant
    for i in range(7, -1, -1):
        if ((round_num + 1) & (1 << i)) != 0:
            circuit.x(key_qubits[i])
    
    return circuit

def encrypt_round(circuit, data_qubits, key_qubits, round_num, is_last_round=False):
    circuit = sub_bytes(circuit, data_qubits)
    circuit.barrier()
    
    circuit = shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    if not is_last_round:
        circuit = mix_columns(circuit, data_qubits)
        circuit.barrier()
    
    circuit = key_expansion(circuit, key_qubits, round_num)
    circuit.barrier()
    
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    
    return circuit

def decrypt_round(circuit, data_qubits, key_qubits, round_num, is_first_round=False):
    circuit = add_round_key(circuit, data_qubits, key_qubits)
    circuit.barrier()
    
    circuit = inverse_key_expansion(circuit, key_qubits, round_num)
    circuit.barrier()
    
    if not is_first_round:
        circuit = inverse_mix_columns(circuit, data_qubits)
        circuit.barrier()
    
    circuit = inverse_shift_rows(circuit, data_qubits)
    circuit.barrier()
    
    circuit = inverse_sub_bytes(circuit, data_qubits)
    
    return circuit

def create_encryption_circuit(plaintext_bits, key_bits, num_rounds=3):
    data_register = QuantumRegister(32, 'data')
    key_register = QuantumRegister(32, 'key')
    classical_register = ClassicalRegister(32, 'c')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    # Initialize plaintext
    for i in range(32):
        if plaintext_bits[i] == '1':
            circuit.x(data_register[i])
    
    # Initialize key
    for i in range(32):
        if key_bits[i] == '1':
            circuit.x(key_register[i])
    
    circuit.barrier()
    
    # Initial round key addition
    circuit = add_round_key(circuit, data_register, key_register)
    
    circuit.barrier()
    
    # Main rounds
    for r in range(num_rounds - 1):
        circuit = encrypt_round(circuit, data_register, key_register, r)
        circuit.barrier()
    
    # Final round (no MixColumns)
    circuit = encrypt_round(circuit, data_register, key_register, num_rounds - 1, is_last_round=True)
    
    circuit.measure(data_register, classical_register)
    
    return circuit

def create_decryption_circuit(ciphertext_bits, key_bits, num_rounds=3):
    data_register = QuantumRegister(32, 'data')
    key_register = QuantumRegister(32, 'key')
    classical_register = ClassicalRegister(32, 'c')
    
    circuit = QuantumCircuit(data_register, key_register, classical_register)
    
    # Initialize ciphertext
    for i in range(32):
        if ciphertext_bits[i] == '1':
            circuit.x(data_register[i])
    
    # Initialize key
    for i in range(32):
        if key_bits[i] == '1':
            circuit.x(key_register[i])
    
    # Generate all round keys
    round_keys = []
    temp_circuit = QuantumCircuit(key_register.copy())
    for r in range(num_rounds):
        temp_circuit = key_expansion(temp_circuit, temp_circuit.qubits, r)
    
    circuit.barrier()
    
    # Inverse rounds
    # Final round (no MixColumns)
    circuit = decrypt_round(circuit, data_register, key_register, num_rounds - 1, is_first_round=True)
    circuit.barrier()
    
    # Main rounds
    for r in range(num_rounds - 2, -1, -1):
        circuit = decrypt_round(circuit, data_register, key_register, r)
        circuit.barrier()
    
    # Initial round key addition
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
    
    print("\nTesting with different message:")
    test_message = "ABCD"
    print(f"Message: {test_message}")
    
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