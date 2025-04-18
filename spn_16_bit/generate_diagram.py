from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def create_grover_spn_circuit_with_boxes():
    data = QuantumRegister(16, 'data')
    key = QuantumRegister(16, 'key')
    target = QuantumRegister(1, 'target')
    cr = ClassicalRegister(16, 'cr')
    
    circuit = QuantumCircuit(data, key, target, cr)
    
    for i, bit in enumerate(reversed("1101101100100101")):
        if bit == '1':
            circuit.x(data[i])
    
    circuit.h(key)
    
    circuit.x(target)
    circuit.h(target)
    
    oracle_circuit = QuantumCircuit(data, key, target)
    
    round1 = QuantumCircuit(data, key)
    round1.barrier()
    
    add_key_box = QuantumCircuit(data, key)
    for i in range(16):
        add_key_box.cx(key[i], data[i])
    add_key_inst = add_key_box.to_instruction(label="Add Round Key")
    round1.append(add_key_inst, data[:] + key[:])
    
    s_box = QuantumCircuit(data)
    s_box.swap(data[0], data[3])
    s_box_inst = s_box.to_instruction(label="S-box Layer")
    round1.append(s_box_inst, data[:])
    
    shift_rows = QuantumCircuit(data)
    shift_rows.swap(data[4], data[12])
    shift_rows_inst = shift_rows.to_instruction(label="Shift Rows")
    round1.append(shift_rows_inst, data[:])
    
    mix_columns = QuantumCircuit(data)
    mix_columns.swap(data[0], data[8])
    mix_columns_inst = mix_columns.to_instruction(label="Mix Columns")
    round1.append(mix_columns_inst, data[:])
    
    key_expansion = QuantumCircuit(key)
    key_expansion.swap(key[0], key[8])
    key_exp_inst = key_expansion.to_instruction(label="Key Expansion")
    round1.append(key_exp_inst, key[:])
    
    round1_inst = round1.to_instruction(label="Round 1")
    oracle_circuit.append(round1_inst, data[:] + key[:])
    
    round2 = QuantumCircuit(data, key)
    round2.barrier()
    
    round2.append(add_key_inst, data[:] + key[:])
    round2.append(s_box_inst, data[:])
    round2.append(shift_rows_inst, data[:])
    round2.append(key_exp_inst, key[:])
    round2.append(add_key_inst, data[:] + key[:])
    
    round2_inst = round2.to_instruction(label="Round 2")
    oracle_circuit.append(round2_inst, data[:] + key[:])
    
    check_output = QuantumCircuit(data, target)
    expected_output = "0011001000110100"
    
    for i, bit in enumerate(reversed(expected_output)):
        if bit == '0':
            check_output.x(data[i])
    
    check_output.mcx(data[:], target[0])
    
    for i, bit in enumerate(reversed(expected_output)):
        if bit == '0':
            check_output.x(data[i])
    
    check_inst = check_output.to_instruction(label="Check Ciphertext")
    oracle_circuit.append(check_inst, data[:] + target[:])
    
    # Create inverse operations
    inv_s_box = QuantumCircuit(data)
    inv_s_box.swap(data[0], data[3])  # Same as forward for this simplified example
    inv_s_box_inst = inv_s_box.to_instruction(label="Inverse S-box")
    
    inv_shift_rows = QuantumCircuit(data)
    inv_shift_rows.swap(data[4], data[12])  # Same as forward for this simplified example
    inv_shift_rows_inst = inv_shift_rows.to_instruction(label="Inverse Shift Rows")
    
    inv_mix_columns = QuantumCircuit(data)
    inv_mix_columns.swap(data[0], data[8])  # Same as forward for this simplified example
    inv_mix_columns_inst = inv_mix_columns.to_instruction(label="Inverse Mix Columns")
    
    inv_key_exp = QuantumCircuit(key)
    inv_key_exp.swap(key[0], key[8])  # Same as forward for this simplified example
    inv_key_exp_inst = inv_key_exp.to_instruction(label="Inverse Key Expansion")
    
    # Uncompute round 2
    uncompute_round2 = QuantumCircuit(data, key)
    uncompute_round2.barrier()
    uncompute_round2.append(add_key_inst, data[:] + key[:])
    uncompute_round2.append(inv_key_exp_inst, key[:])
    uncompute_round2.append(inv_shift_rows_inst, data[:])
    uncompute_round2.append(inv_s_box_inst, data[:])
    uncompute_round2.append(add_key_inst, data[:] + key[:])
    
    uncompute_round2_inst = uncompute_round2.to_instruction(label="Inverse Round 2")
    oracle_circuit.append(uncompute_round2_inst, data[:] + key[:])
    
    # Uncompute round 1
    uncompute_round1 = QuantumCircuit(data, key)
    uncompute_round1.barrier()
    uncompute_round1.append(inv_key_exp_inst, key[:])
    uncompute_round1.append(inv_mix_columns_inst, data[:])
    uncompute_round1.append(inv_shift_rows_inst, data[:])
    uncompute_round1.append(inv_s_box_inst, data[:])
    uncompute_round1.append(add_key_inst, data[:] + key[:])
    
    uncompute_round1_inst = uncompute_round1.to_instruction(label="Inverse Round 1")
    oracle_circuit.append(uncompute_round1_inst, data[:] + key[:])
    
    oracle_inst = oracle_circuit.to_instruction(label="Oracle")
    circuit.append(oracle_inst, data[:] + key[:] + target[:])
    
    diffusion = QuantumCircuit(key)
    diffusion.barrier()
    
    diffusion.h(key)
    diffusion.x(key)
    diffusion.h(key[-1])
    diffusion.mcx(key[:-1], key[-1])
    diffusion.h(key[-1])
    diffusion.x(key)
    diffusion.h(key)
    
    diffusion_inst = diffusion.to_instruction(label="Grover Diffusion")
    circuit.append(diffusion_inst, key[:])
    
    circuit.measure(key, cr)
    
    return circuit

circuit = create_grover_spn_circuit_with_boxes()

fig = plt.figure(figsize=(20, 50))
circuit_drawer(circuit, output='mpl', style={'name': 'bw'}, fold=90)
plt.savefig('grover_spn_circuit.png', dpi=150, bbox_inches='tight')

print(f"Circuit depth: {circuit.depth()}")
print(f"Circuit width: {circuit.num_qubits}")
print(f"Total gates: {sum(circuit.count_ops().values())}")

circuit