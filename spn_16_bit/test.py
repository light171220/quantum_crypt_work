from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

def universal_mcx_gate(circuit, control_qubits, target_qubit):
    num_controls = len(control_qubits)
    
    if num_controls == 0:
        return circuit
    
    if num_controls == 1:
        circuit.cx(control_qubits[0], target_qubit)
        return circuit
    
    if num_controls == 2:
        circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        return circuit
    
    def recursive_mcx(controls, target):
        if len(controls) <= 2:
            circuit.mcx(controls, target)
            return
        
        if len(controls) == 3:
            circuit.ccx(controls[0], controls[1], controls[2])
            circuit.cx(controls[2], target)
            circuit.ccx(controls[0], controls[1], controls[2])
            return
        
        # Recursive approach for more than 3 controls
        # 1. Create an ancilla qubit
        ancilla = circuit.qregs[0][-1]
        
        # 2. Recursively control the first set of controls on an ancilla
        first_half = controls[:len(controls)//2]
        second_half = controls[len(controls)//2:]
        
        # Recursively apply multi-controlled gate to first half
        recursive_mcx(first_half, ancilla)
        
        # Recursively apply multi-controlled gate to second half
        recursive_mcx(second_half, ancilla)
        
        # Final controlled-X to target
        circuit.cx(ancilla, target)
        
        # Uncompute the ancilla
        recursive_mcx(second_half, ancilla)
        recursive_mcx(first_half, ancilla)
    
    # Call the recursive implementation
    recursive_mcx(control_qubits, target_qubit)
    
    return circuit

def test_universal_mcx(num_controls):
    # Create a quantum circuit with extra ancilla qubits
    qr = QuantumRegister(num_controls + 2)  # Controls + target + ancilla
    cr = ClassicalRegister(num_controls + 1)
    circuit = QuantumCircuit(qr, cr)
    
    # Prepare control qubits in superposition
    for i in range(num_controls):
        circuit.h(qr[i])
    
    # Apply universal MCX
    universal_mcx_gate(circuit, qr[:num_controls], qr[num_controls])
    
    # Measure all qubits
    circuit.measure(qr[:num_controls] + [qr[num_controls]], cr)
    
    # Run the circuit
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts()
    
    print(f"\nResults for {num_controls} control qubits:")
    for outcome, count in counts.items():
        print(f"{outcome}: {count}")
    
    return circuit

def run_mcx_tests():
    # Test with different numbers of control qubits
    test_cases = [1, 2, 3, 4, 5]
    
    for num_controls in test_cases:
        test_universal_mcx(num_controls)

if __name__ == "__main__":
    run_mcx_tests()