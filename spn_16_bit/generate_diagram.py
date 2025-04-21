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
    qc.swap(4, 12)
    qc.swap(5, 13)
    qc.swap(6, 14)
    qc.swap(7, 15)

def inv_shift_rows_function(qc):
    qc.swap(7, 15)
    qc.swap(6, 14)
    qc.swap(5, 13)
    qc.swap(4, 12)

def add_key_function(qc):
    for i in range(16):
        qc.cx(i+16, i)

def key_expansion1_function(qc):
    qc.x(0)
    for i in range(4):
        qc.swap(i+8, i+12)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    qc.append(sbox_gate, [8, 9, 10, 11])
    qc.append(sbox_gate, [12, 13, 14, 15])
    
    for i in range(8):
        qc.cx(i+8, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    qc.append(inv_sbox_gate, [8, 9, 10, 11])
    qc.append(inv_sbox_gate, [12, 13, 14, 15])
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    for i in range(8):
        qc.cx(i, i+8)

def key_expansion2_function(qc):
    qc.x(4)
    qc.x(5)
    for i in range(4):
        qc.swap(i+8, i+12)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    qc.append(sbox_gate, [8, 9, 10, 11])
    qc.append(sbox_gate, [12, 13, 14, 15])
    
    for i in range(8):
        qc.cx(i+8, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    qc.append(inv_sbox_gate, [8, 9, 10, 11])
    qc.append(inv_sbox_gate, [12, 13, 14, 15])
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    for i in range(8):
        qc.cx(i, i+8)

def inv_key_expansion1_function(qc):
    for i in range(8):
        qc.cx(i, i+8)
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    qc.append(sbox_gate, [8, 9, 10, 11])
    qc.append(sbox_gate, [12, 13, 14, 15])
    
    for i in range(8):
        qc.cx(i+8, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    qc.append(inv_sbox_gate, [8, 9, 10, 11])
    qc.append(inv_sbox_gate, [12, 13, 14, 15])
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    qc.x(0)

def inv_key_expansion2_function(qc):
    for i in range(8):
        qc.cx(i, i+8)
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    sub_circuit = QuantumCircuit(4)
    sbox_function(sub_circuit)
    sbox_gate = sub_circuit.to_gate()
    sbox_gate.name = "S-box"
    
    qc.append(sbox_gate, [8, 9, 10, 11])
    qc.append(sbox_gate, [12, 13, 14, 15])
    
    for i in range(8):
        qc.cx(i+8, i)
    
    inv_sub_circuit = QuantumCircuit(4)
    inv_sbox_function(inv_sub_circuit)
    inv_sbox_gate = inv_sub_circuit.to_gate()
    inv_sbox_gate.name = "S-box⁻¹"
    
    qc.append(inv_sbox_gate, [8, 9, 10, 11])
    qc.append(inv_sbox_gate, [12, 13, 14, 15])
    
    for i in range(4):
        qc.swap(i+8, i+12)
    
    qc.x(4)
    qc.x(5)

def diffusion_function(qc):
    for i in range(16):
        qc.h(i)
    
    for i in range(16):
        qc.x(i)
    
    qc.h(15)
    qc.mcx(list(range(15)), 15)
    qc.h(15)
    
    for i in range(16):
        qc.x(i)
    
    for i in range(16):
        qc.h(i)

def create_boxed_spn_circuit():
    key_register = QuantumRegister(16, 'key')
    data_register = QuantumRegister(16, 'data')
    target_register = QuantumRegister(1, 'target')
    classical_register = ClassicalRegister(16, 'classical')
    
    circuit = QuantumCircuit(data_register, key_register, target_register, classical_register)
    
    plaintext = "1101101100100101"
    for i in range(16):
        if plaintext[15-i] == '1':
            circuit.x(data_register[i])
    
    for i in range(16):
        circuit.h(key_register[i])
    
    circuit.x(target_register[0])
    circuit.h(target_register[0])
    
    circuit.barrier(label="Superposition")
    
    sbox_gate = create_custom_gate("S-box", 4, sbox_function)
    inv_sbox_gate = create_custom_gate("S-box⁻¹", 4, inv_sbox_function)
    shift_rows_gate = create_custom_gate("ShiftRows", 16, shift_rows_function)
    inv_shift_rows_gate = create_custom_gate("ShiftRows⁻¹", 16, inv_shift_rows_function)
    mix_cols_gate = create_custom_gate("MixCols", 8, mix_columns_function)
    inv_mix_cols_gate = create_custom_gate("MixCols⁻¹", 8, inv_mix_columns_function)
    
    add_key_circuit = QuantumCircuit(32)
    add_key_function(add_key_circuit)
    add_key_gate = add_key_circuit.to_gate()
    add_key_gate.name = "AddKey"
    
    key_exp1_circuit = QuantumCircuit(16)
    key_expansion1_function(key_exp1_circuit)
    key_exp1_gate = key_exp1_circuit.to_gate()
    key_exp1_gate.name = "KeyExp1"
    
    key_exp2_circuit = QuantumCircuit(16)
    key_expansion2_function(key_exp2_circuit)
    key_exp2_gate = key_exp2_circuit.to_gate()
    key_exp2_gate.name = "KeyExp2"
    
    inv_key_exp1_circuit = QuantumCircuit(16)
    inv_key_expansion1_function(inv_key_exp1_circuit)
    inv_key_exp1_gate = inv_key_exp1_circuit.to_gate()
    inv_key_exp1_gate.name = "KeyExp1⁻¹"
    
    inv_key_exp2_circuit = QuantumCircuit(16)
    inv_key_expansion2_function(inv_key_exp2_circuit)
    inv_key_exp2_gate = inv_key_exp2_circuit.to_gate()
    inv_key_exp2_gate.name = "KeyExp2⁻¹"
    
    diffusion_gate = create_custom_gate("Diffusion", 16, diffusion_function)
    
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(sbox_gate, [data_register[0], data_register[1], data_register[2], data_register[3]])
    circuit.append(sbox_gate, [data_register[4], data_register[5], data_register[6], data_register[7]])
    circuit.append(sbox_gate, [data_register[8], data_register[9], data_register[10], data_register[11]])
    circuit.append(sbox_gate, [data_register[12], data_register[13], data_register[14], data_register[15]])
    circuit.append(shift_rows_gate, data_register[:])
    circuit.append(mix_cols_gate, data_register[0:8])
    circuit.append(mix_cols_gate, data_register[8:16])
    circuit.append(key_exp1_gate, key_register[:])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(sbox_gate, [data_register[0], data_register[1], data_register[2], data_register[3]])
    circuit.append(sbox_gate, [data_register[4], data_register[5], data_register[6], data_register[7]])
    circuit.append(sbox_gate, [data_register[8], data_register[9], data_register[10], data_register[11]])
    circuit.append(sbox_gate, [data_register[12], data_register[13], data_register[14], data_register[15]])
    circuit.append(shift_rows_gate, data_register[:])
    circuit.append(key_exp2_gate, key_register[:])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    circuit.barrier()
    
    ciphertext = "0010110001010101"
    for i in range(16):
        if ciphertext[15-i] == '0':
            circuit.x(data_register[i])
    
    circuit.mcx(data_register[:], target_register[0])
    
    for i in range(16):
        if ciphertext[15-i] == '0':
            circuit.x(data_register[i])
            
    circuit.barrier()
    
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(inv_key_exp2_gate, key_register[:])
    circuit.append(inv_shift_rows_gate, data_register[:])
    circuit.append(inv_sbox_gate, [data_register[12], data_register[13], data_register[14], data_register[15]])
    circuit.append(inv_sbox_gate, [data_register[8], data_register[9], data_register[10], data_register[11]])
    circuit.append(inv_sbox_gate, [data_register[4], data_register[5], data_register[6], data_register[7]])
    circuit.append(inv_sbox_gate, [data_register[0], data_register[1], data_register[2], data_register[3]])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    circuit.append(inv_key_exp1_gate, key_register[:])
    circuit.append(inv_mix_cols_gate, data_register[8:16])
    circuit.append(inv_mix_cols_gate, data_register[0:8])
    circuit.append(inv_shift_rows_gate, data_register[:])
    circuit.append(inv_sbox_gate, [data_register[12], data_register[13], data_register[14], data_register[15]])
    circuit.append(inv_sbox_gate, [data_register[8], data_register[9], data_register[10], data_register[11]])
    circuit.append(inv_sbox_gate, [data_register[4], data_register[5], data_register[6], data_register[7]])
    circuit.append(inv_sbox_gate, [data_register[0], data_register[1], data_register[2], data_register[3]])
    circuit.append(add_key_gate, data_register[:] + key_register[:])
    
    circuit.barrier(label="Oracle Complete")
    
    circuit.append(diffusion_gate, key_register)
    
    circuit.barrier(label="Diffusion Complete")
    
    circuit.measure(key_register, classical_register)
    
    output_dir = "diagrams"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(24, 16))
    circuit.draw(output='mpl', 
                 style={'backgroundcolor': '#EEEEEE',
                        'linecolor': 'black',
                        'textcolor': 'black',
                        'gatefacecolor': 'white',
                        'gatetextcolor': 'black'},
                 fold=70,
                 scale=0.7)
    plt.title("16-bit SPN Block Cipher Circuit - Grover's Algorithm", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spn_boxed_circuit.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxed SPN circuit diagram saved as '{output_dir}/spn_boxed_circuit.png'")
    return circuit

if __name__ == "__main__":
    print("\nGenerating boxed SPN block cipher circuit diagram...")
    boxed_circuit = create_boxed_spn_circuit()
    print("Circuit diagram generated successfully!")