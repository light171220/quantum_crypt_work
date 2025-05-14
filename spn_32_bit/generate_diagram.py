from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit.circuit import Gate
import matplotlib.pyplot as plt
import os

def create_custom_gate(name, num_qubits, circuit_function):
    qc = QuantumCircuit(num_qubits)
    circuit_function(qc)
    gate = qc.to_gate()
    gate.name = name
    return gate

def sbox_function(qc):
    qc.swap(1, 2)
    qc.swap(0, 3)
    qc.cx(2, 1)
    qc.ccx(3, 1, 0)
    qc.ccx(0, 2, 3)
    qc.ccx(0, 3, 1)
    qc.mcx([0, 1, 3], 2)
    qc.cx(3, 0)
    qc.x(2)
    qc.x(3)
    qc.cx(0, 2)
    qc.ccx(2, 1, 0)
    qc.ccx(0, 3, 1)
    qc.ccx(0, 1, 3)
    qc.swap(0, 1)
    qc.swap(0, 2)
    qc.swap(1, 2)

def inv_sbox_function(qc):
    qc.swap(1, 2)
    qc.swap(0, 2)
    qc.swap(0, 1)
    qc.ccx(0, 1, 3)
    qc.ccx(0, 3, 1)
    qc.ccx(2, 1, 0)
    qc.cx(0, 2)
    qc.x(3)
    qc.x(2)
    qc.cx(3, 0)
    qc.mcx([0, 1, 3], 2)
    qc.ccx(0, 3, 1)
    qc.ccx(0, 2, 3)
    qc.ccx(3, 1, 0)
    qc.cx(2, 1)
    qc.swap(0, 3)
    qc.swap(1, 2)

def mix_columns_function(qc):
    qc.cx(0, 6)
    qc.cx(5, 3)
    qc.cx(4, 2)
    qc.cx(1, 7)
    qc.cx(7, 4)
    qc.cx(2, 5)
    qc.cx(3, 0)
    qc.cx(6, 1)
    qc.swap(0, 2)
    qc.swap(1, 4)
    qc.swap(2, 5)
    qc.swap(4, 6)

def inv_mix_columns_function(qc):
    qc.swap(4, 6)
    qc.swap(2, 5)
    qc.swap(1, 4)
    qc.swap(0, 2)
    qc.cx(6, 1)
    qc.cx(3, 0)
    qc.cx(2, 5)
    qc.cx(7, 4)
    qc.cx(1, 7)
    qc.cx(4, 2)
    qc.cx(5, 3)
    qc.cx(0, 6)

def shift_rows_function(qc):
    # Expanded for 32-bit - modified to handle 8 rows of 4 bits each
    qc.swap(4, 12)
    qc.swap(5, 13)
    qc.swap(6, 14)
    qc.swap(7, 15)
    
    # Additional shifts for expanded rows
    qc.swap(20, 28)
    qc.swap(21, 29)
    qc.swap(22, 30)
    qc.swap(23, 31)
    
    # Middle row shifts (for pattern consistency)
    qc.swap(16, 24)
    qc.swap(17, 25)
    qc.swap(18, 26)
    qc.swap(19, 27)
    
    # Optional additional mixing
    qc.swap(8, 24)
    qc.swap(9, 25)
    qc.swap(10, 26)
    qc.swap(11, 27)

def inv_shift_rows_function(qc):
    # Inverse of expanded shift rows
    qc.swap(11, 27)
    qc.swap(10, 26)
    qc.swap(9, 25)
    qc.swap(8, 24)
    
    qc.swap(19, 27)
    qc.swap(18, 26)
    qc.swap(17, 25)
    qc.swap(16, 24)
    
    qc.swap(23, 31)
    qc.swap(22, 30)
    qc.swap(21, 29)
    qc.swap(20, 28)
    
    qc.swap(7, 15)
    qc.swap(6, 14)
    qc.swap(5, 13)
    qc.swap(4, 12)

def add_key_function(qc):
    for i in range(32):
        qc.cx(i+32, i)

def key_expansion1_function(qc):
    qc.x(0)
    for i in range(8):
        qc.swap(i+16, i+24)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    # Apply S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(16):
        qc.cx(i+16, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    # Apply inverse S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(inv_sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    for i in range(16):
        qc.cx(i, i+16)

def key_expansion2_function(qc):
    qc.x(4)
    qc.x(5)
    for i in range(8):
        qc.swap(i+16, i+24)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    # Apply S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(16):
        qc.cx(i+16, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    # Apply inverse S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(inv_sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    for i in range(16):
        qc.cx(i, i+16)

def inv_key_expansion1_function(qc):
    for i in range(16):
        qc.cx(i, i+16)
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    # Apply S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(16):
        qc.cx(i+16, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    # Apply inverse S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(inv_sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    qc.x(0)

def inv_key_expansion2_function(qc):
    for i in range(16):
        qc.cx(i, i+16)
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    # Apply S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(16):
        qc.cx(i+16, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    # Apply inverse S-box to each 4-qubit block within bounds
    for i in range(0, 16, 4):
        qc.append(inv_sbox_gate, [i+16, i+17, i+18, i+19])
    
    for i in range(8):
        qc.swap(i+16, i+24)
    
    qc.x(4)
    qc.x(5)

def diffusion_function(qc):
    for i in range(32):
        qc.h(i)
    
    for i in range(32):
        qc.x(i)
    
    qc.h(31)
    qc.mcx(list(range(31)), 31)
    qc.h(31)
    
    for i in range(32):
        qc.x(i)
    
    for i in range(32):
        qc.h(i)

def create_boxed_spn_circuit():
    key_register = QuantumRegister(32, 'key')
    data_register = QuantumRegister(32, 'data')
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(32, 'classical')
    
    circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)
    
    plaintext = "11011011001001011010100110010110" + "10101101100100101101011001001010"
    for i in range(32):
        if plaintext[31-i] == '1':
            circuit.x(data_register[i])
    
    for i in range(32):
        circuit.h(key_register[i])
    
    circuit.x(target_register[0])
    circuit.h(target_register[0])
    
    circuit.barrier(label="Superposition")
    
    sbox_gate = create_custom_gate("S-box", 4, sbox_function)
    inv_sbox_gate = create_custom_gate("S-box⁻¹", 4, inv_sbox_function)
    shift_rows_gate = create_custom_gate("ShiftRows", 32, shift_rows_function)
    inv_shift_rows_gate = create_custom_gate("ShiftRows⁻¹", 32, inv_shift_rows_function)
    mix_cols_gate = create_custom_gate("MixCols", 8, mix_columns_function)
    inv_mix_cols_gate = create_custom_gate("MixCols⁻¹", 8, inv_mix_columns_function)
    
    add_key_circuit = QuantumCircuit(64)
    add_key_function(add_key_circuit)
    add_key_gate = add_key_circuit.to_gate()
    add_key_gate.name = "AddKey"
    
    key_exp1_circuit = QuantumCircuit(32)
    key_expansion1_function(key_exp1_circuit)
    key_exp1_gate = key_exp1_circuit.to_gate()
    key_exp1_gate.name = "KeyExp1"
    
    key_exp2_circuit = QuantumCircuit(32)
    key_expansion2_function(key_exp2_circuit)
    key_exp2_gate = key_exp2_circuit.to_gate()
    key_exp2_gate.name = "KeyExp2"
    
    inv_key_exp1_circuit = QuantumCircuit(32)
    inv_key_expansion1_function(inv_key_exp1_circuit)
    inv_key_exp1_gate = inv_key_exp1_circuit.to_gate()
    inv_key_exp1_gate.name = "KeyExp1⁻¹"
    
    inv_key_exp2_circuit = QuantumCircuit(32)
    inv_key_expansion2_function(inv_key_exp2_circuit)
    inv_key_exp2_gate = inv_key_exp2_circuit.to_gate()
    inv_key_exp2_gate.name = "KeyExp2⁻¹"
    
    diffusion_gate = create_custom_gate("Diffusion", 32, diffusion_function)
    
    # Forward encryption path
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    for i in range(0, 32, 4):
        circuit.append(sbox_gate, [data_register[i], data_register[i+1], 
                                   data_register[i+2], data_register[i+3]])
    
    circuit.append(shift_rows_gate, data_register[:])
    
    # Apply mix columns to each 8-qubit block
    for i in range(0, 32, 8):
        if i + 8 <= 32:  # Ensure we don't go out of bounds
            circuit.append(mix_cols_gate, data_register[i:i+8])
    
    circuit.append(key_exp1_gate, key_register[:])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    for i in range(0, 32, 4):
        circuit.append(sbox_gate, [data_register[i], data_register[i+1], 
                                   data_register[i+2], data_register[i+3]])
    
    circuit.append(shift_rows_gate, data_register[:])
    circuit.append(key_exp2_gate, key_register[:])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    circuit.barrier()
    
    ciphertext = "00101100010101010110110001010101" + "01100110010101010110011001010101"
    for i in range(32):
        if ciphertext[31-i] == '0':
            circuit.x(data_register[i])
    
    circuit.mcx(data_register[:], target_register[0])
    
    for i in range(32):
        if ciphertext[31-i] == '0':
            circuit.x(data_register[i])
            
    circuit.barrier()
    
    # Reverse decryption path
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(inv_key_exp2_gate, key_register[:])
    circuit.append(inv_shift_rows_gate, data_register[:])
    
    for i in range(0, 32, 4):
        circuit.append(inv_sbox_gate, [data_register[i], data_register[i+1], 
                                       data_register[i+2], data_register[i+3]])
    
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(inv_key_exp1_gate, key_register[:])
    
    # Apply inverse mix columns to each 8-qubit block
    for i in range(0, 32, 8):
        if i + 8 <= 32:  # Ensure we don't go out of bounds
            circuit.append(inv_mix_cols_gate, data_register[i:i+8])
    
    circuit.append(inv_shift_rows_gate, data_register[:])
    
    for i in range(0, 32, 4):
        circuit.append(inv_sbox_gate, [data_register[i], data_register[i+1], 
                                       data_register[i+2], data_register[i+3]])
    
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    circuit.barrier(label="Oracle Complete")
    
    circuit.append(diffusion_gate, key_register)
    
    circuit.barrier(label="Diffusion Complete")
    
    circuit.measure(key_register, classical_register)
    
    output_dir = "diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(30, 20))
    circuit.draw(output='mpl', 
                 style={'backgroundcolor': '#EEEEEE',
                        'linecolor': 'black',
                        'textcolor': 'black',
                        'gatefacecolor': 'white',
                        'gatetextcolor': 'black'},
                 fold=90,
                 scale=0.65)
    plt.title("32-bit SPN Block Cipher Circuit - Grover's Algorithm", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spn_boxed_circuit_32bit.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxed SPN circuit diagram saved as '{output_dir}/spn_boxed_circuit_32bit.png'")
    return circuit

if __name__ == "__main__":
    print("\nGenerating 32-bit boxed SPN block cipher circuit diagram...")
    boxed_circuit = create_boxed_spn_circuit()
    print("Circuit diagram generated successfully!")