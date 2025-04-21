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
    qc.cx(2, 1)
    qc.x(0)
    qc.ccx(1, 2, 3)
    qc.ccx(1, 3, 2)
    qc.cx(3, 1)
    qc.x(3)
    qc.ccx(2, 0, 1)
    qc.cx(0, 2)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.id(0)
    qc.id(0)
    qc.ccx(1, 2, 3)
    qc.id(0)
    qc.swap(2, 3)

def inv_sbox_function(qc):
    qc.swap(2, 3)
    qc.id(0)
    qc.ccx(1, 2, 3)
    qc.id(0)
    qc.id(0)
    qc.cx(1, 2)
    qc.cx(3, 0)
    qc.cx(0, 2)
    qc.ccx(2, 0, 1)
    qc.x(3)
    qc.cx(3, 1)
    qc.ccx(1, 3, 2)
    qc.ccx(1, 2, 3)
    qc.x(0)
    qc.cx(2, 1)

def permutation_function(qc):
    qc.swap(7, 0)
    qc.swap(6, 3)
    qc.swap(6, 5)
    qc.swap(4, 2)
    qc.swap(4, 1)

def inv_permutation_function(qc):
    qc.swap(4, 1)
    qc.swap(4, 2)
    qc.swap(6, 5)
    qc.swap(6, 3)
    qc.swap(7, 0)

def key_gen_function(qc):
    qc.cx(2, 1)
    qc.x(0)
    qc.ccx(1, 2, 3)
    qc.ccx(1, 3, 2)
    qc.cx(3, 1)
    qc.x(3)
    qc.ccx(2, 0, 1)
    qc.cx(0, 2)
    qc.cx(3, 0)
    qc.cx(1, 2)
    qc.id(0)
    qc.id(0)
    qc.ccx(1, 2, 3)
    qc.id(0)
    qc.swap(2, 3)
    
    qc.cx(6, 5)
    qc.x(4)
    qc.ccx(5, 6, 7)
    qc.ccx(5, 7, 6)
    qc.cx(7, 5)
    qc.x(7)
    qc.ccx(6, 4, 5)
    qc.cx(4, 6)
    qc.cx(7, 4)
    qc.cx(5, 6)
    qc.id(4)
    qc.id(4)
    qc.ccx(5, 6, 7)
    qc.id(4)
    qc.swap(6, 7)
    
    qc.swap(0, 3)
    qc.swap(2, 3)
    qc.swap(1, 3)
    qc.swap(4, 7)
    qc.swap(6, 7)
    qc.swap(5, 7)

def inv_key_gen_function(qc):
    qc.swap(5, 7)
    qc.swap(6, 7)
    qc.swap(4, 7)
    qc.swap(1, 3)
    qc.swap(2, 3)
    qc.swap(0, 3)
    
    qc.swap(6, 7)
    qc.id(4)
    qc.ccx(5, 6, 7)
    qc.id(4)
    qc.id(4)
    qc.cx(5, 6)
    qc.cx(7, 4)
    qc.cx(4, 6)
    qc.ccx(6, 4, 5)
    qc.x(7)
    qc.cx(7, 5)
    qc.ccx(5, 7, 6)
    qc.ccx(5, 6, 7)
    qc.x(4)
    qc.cx(6, 5)
    
    qc.swap(2, 3)
    qc.id(0)
    qc.ccx(1, 2, 3)
    qc.id(0)
    qc.id(0)
    qc.cx(1, 2)
    qc.cx(3, 0)
    qc.cx(0, 2)
    qc.ccx(2, 0, 1)
    qc.x(3)
    qc.cx(3, 1)
    qc.ccx(1, 3, 2)
    qc.ccx(1, 2, 3)
    qc.x(0)
    qc.cx(2, 1)

def diffusion_function(qc):
    for i in range(8):
        qc.h(i)
    
    for i in range(8):
        qc.x(i)
    
    qc.h(7)
    qc.mcx(list(range(7)), 7)
    qc.h(7)
    
    for i in range(8):
        qc.x(i)
    
    for i in range(8):
        qc.h(i)

def add_key_function(qc):
    for i in range(8):
        qc.cx(i+8, i)

def inv_add_key_function(qc):
    for i in range(8):
        qc.cx(i+8, i)

def oracle_function(qc):
    ciphertext = "00100010"
    for i in range(8):
        if ciphertext[7-i] == '0':
            qc.x(i)
    
    qc.mcx(list(range(8)), 16)
    
    for i in range(8):
        if ciphertext[7-i] == '0':
            qc.x(i)

def create_boxed_yoyo_circuit():
    key_reg = QuantumRegister(8, 'key')
    pt_reg = QuantumRegister(8, 'pt')
    target_reg = QuantumRegister(1, 'target')
    cr = ClassicalRegister(8, 'cr')
    
    circuit = QuantumCircuit(pt_reg, key_reg, target_reg, cr)
    
    plaintext = "11011011"
    for i in range(8):
        if plaintext[7-i] == '1':
            circuit.x(pt_reg[i])
    
    for i in range(8):
        circuit.h(key_reg[i])
    
    circuit.x(target_reg[0])
    circuit.h(target_reg[0])
    
    circuit.barrier()
    
    sbox_gate = create_custom_gate("S-box", 4, sbox_function)
    inv_sbox_gate = create_custom_gate("S-box⁻¹", 4, inv_sbox_function)
    perm_gate = create_custom_gate("Perm", 8, permutation_function)
    inv_perm_gate = create_custom_gate("Perm⁻¹", 8, inv_permutation_function)
    key_gen_gate = create_custom_gate("KeyGen", 8, key_gen_function)
    inv_key_gen_gate = create_custom_gate("KeyGen⁻¹", 8, inv_key_gen_function)
    diffusion_gate = create_custom_gate("Diffusion", 8, diffusion_function)
    
    add_key_circuit = QuantumCircuit(16)
    add_key_function(add_key_circuit)
    add_key_gate = add_key_circuit.to_gate()
    add_key_gate.name = "AddKey"
    
    inv_add_key_circuit = QuantumCircuit(16)
    inv_add_key_function(inv_add_key_circuit)
    inv_add_key_gate = inv_add_key_circuit.to_gate()
    inv_add_key_gate.name = "AddKey⁻¹"
    
    
    oracle_circuit = QuantumCircuit(17)
    oracle_function(oracle_circuit)
    oracle_gate = oracle_circuit.to_gate()
    oracle_gate.name = "Oracle"

    
    circuit.append(add_key_gate, pt_reg[:] + key_reg[:])
    circuit.append(sbox_gate, [pt_reg[0], pt_reg[1], pt_reg[2], pt_reg[3]])
    circuit.append(sbox_gate, [pt_reg[4], pt_reg[5], pt_reg[6], pt_reg[7]])
    circuit.append(perm_gate, pt_reg)
    circuit.append(key_gen_gate, key_reg)
    circuit.append(add_key_gate, pt_reg[:] + key_reg[:])
    circuit.append(sbox_gate, [pt_reg[0], pt_reg[1], pt_reg[2], pt_reg[3]])
    circuit.append(sbox_gate, [pt_reg[4], pt_reg[5], pt_reg[6], pt_reg[7]])
    circuit.append(perm_gate, pt_reg)
    circuit.append(oracle_gate, pt_reg[:] + key_reg[:] + target_reg[:])
    circuit.append(inv_perm_gate, pt_reg)
    circuit.append(inv_sbox_gate, [pt_reg[0], pt_reg[1], pt_reg[2], pt_reg[3]])
    circuit.append(inv_sbox_gate, [pt_reg[4], pt_reg[5], pt_reg[6], pt_reg[7]])
    circuit.append(inv_add_key_gate, pt_reg[:] + key_reg[:])
    circuit.append(inv_key_gen_gate, key_reg)
    circuit.append(inv_perm_gate, pt_reg)
    circuit.append(inv_sbox_gate, [pt_reg[0], pt_reg[1], pt_reg[2], pt_reg[3]])
    circuit.append(inv_sbox_gate, [pt_reg[4], pt_reg[5], pt_reg[6], pt_reg[7]])
    circuit.append(inv_add_key_gate, pt_reg[:] + key_reg[:])
    circuit.append(diffusion_gate, key_reg)
    
    circuit.barrier()
    
    circuit.measure(key_reg, cr)
    
    output_dir = "quantum_iterations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(24, 16))
    circuit.draw(output='mpl', 
                 style={'backgroundcolor': '#EEEEEE',
                        'linecolor': 'black',
                        'textcolor': 'black',
                        'gatefacecolor': 'white',
                        'gatetextcolor': 'black'},
                 fold=40,
                 scale=0.5)
    plt.title("Yo-yo Block Cipher Circuit - Grover's Algorithm", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/yoyo_boxed_circuit.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxed Yo-yo circuit diagram saved as '{output_dir}/yoyo_boxed_circuit.png'")
    return circuit

if __name__ == "__main__":
    print("\nGenerating boxed Yo-yo block cipher circuit diagram...")
    boxed_circuit = create_boxed_yoyo_circuit()
    print("Circuit diagram generated successfully!")