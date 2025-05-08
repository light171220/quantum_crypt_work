from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

def s0_box_circuit(circuit, qubits):
    [q0, q1, q2, q3] = qubits
    
    circuit.ccx(q0, q2, q3)
    circuit.ccx(q0, q1, q3)
    circuit.swap(q1, q3)
    circuit.x(q2)
    circuit.ccx(q0, q1, q2)
    circuit.cx(q3, q1)
    circuit.cx(q2, q0)
    
    circuit.swap(q0, q1)
    
    return circuit

def permute(bits, pattern):
    return [bits[i-1] for i in pattern]

def bits_to_str(bits):
    return ''.join(map(str, bits))

def s_box_classical(bits, box):
    row = bits[0] * 2 + bits[3]
    col = bits[1] * 2 + bits[2]
    return [int(x) for x in f"{box[row][col]:02b}"]

def verify_s0_box():
    s0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 2, 0]
    ]
    
    all_results = {}
    for i in range(16):
        input_bits = [int(b) for b in format(i, '04b')]
        
        row = input_bits[0] * 2 + input_bits[3]
        col = input_bits[1] * 2 + input_bits[2]
        s0_value = s0[row][col]
        s0_bits = [int(b) for b in format(s0_value, '02b')]
        
        qreg = QuantumRegister(4, 'q')
        creg = ClassicalRegister(2, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        for j in range(4):
            if input_bits[j] == 1:
                circuit.x(qreg[j])
        
        circuit = s0_box_circuit(circuit, qreg)
        
        circuit.measure(qreg[0], creg[0])
        circuit.measure(qreg[1], creg[1])
        
        if input_bits == [1, 0, 1, 0]:
            circuit.draw(output='mpl', filename='s0_box_circuit_1010.png')
        
        simulator = AerSimulator()
        job = simulator.run(circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        most_frequent = max(counts, key=counts.get)
        quantum_result = [int(most_frequent[1-i]) for i in range(2)]
        
        input_str = bits_to_str(input_bits)
        classical_str = bits_to_str(s0_bits)
        quantum_str = bits_to_str(quantum_result)
        match = (s0_bits == quantum_result)
        
        all_results[input_str] = {
            'classical': classical_str,
            'quantum': quantum_str,
            'match': match
        }
        
        if input_bits == [1, 0, 1, 0]:
            plot_histogram(counts, title="S0 Box Results for input 1010").savefig('s0_box_histogram_1010.png')
    
    print("S0 Box Test Results")
    print("-" * 40)
    print("| Input | Classical | Quantum | Match |")
    print("-" * 40)
    
    all_match = True
    for input_str, result in all_results.items():
        match_str = "✓" if result['match'] else "✗"
        print(f"| {input_str} | {result['classical']} | {result['quantum']} | {match_str} |")
        if not result['match']:
            all_match = False
    
    print("-" * 40)
    print(f"Overall Match: {'PASS' if all_match else 'FAIL'}")
    
    example = all_results['1010']
    print("\nDetailed analysis for input 1010:")
    print(f"Classical output: {example['classical']}")
    print(f"Quantum output: {example['quantum']}")
    print(f"Match: {example['match']}")
    
    return all_match

if __name__ == "__main__":
    success = verify_s0_box()
    print(f"S0 Box Verification: {'PASS' if success else 'FAIL'}")