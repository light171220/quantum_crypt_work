from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
import time
import random

aes_sbox = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

class AdvancedReversibleSboxBuilder:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.quantum_cost = 0
        self.gate_count = 0
        self.two_qubit_cost = 0
        self.simulator = AerSimulator()
    
    def multi_controlled_toffoli(self, circuit, control_qubits, target_qubit):
        if len(control_qubits) == 1:
            circuit.cx(control_qubits[0], target_qubit)
        elif len(control_qubits) == 2:
            circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
        else:
            temp_anc = circuit.qubits[len(circuit.qubits)-1]
            circuit.ccx(control_qubits[0], control_qubits[1], temp_anc)
            for i in range(2, len(control_qubits)):
                circuit.ccx(control_qubits[i], temp_anc, target_qubit)
                circuit.ccx(control_qubits[0], control_qubits[1], temp_anc)
    
    def build_gate_set(self):
        return [
            {'name': 'NOT', 'qubits': 1, 'controls': 0, 'quantum_cost': 1, 'two_qubit_cost': 0, 'gate_count': 1},
            {'name': 'CNOT', 'qubits': 2, 'controls': 1, 'quantum_cost': 1, 'two_qubit_cost': 1, 'gate_count': 1},
            {'name': 'Toffoli', 'qubits': 3, 'controls': 2, 'quantum_cost': 5, 'two_qubit_cost': 5, 'gate_count': 1},
            {'name': 'controlled_Toffoli', 'qubits': 4, 'controls': 3, 'quantum_cost': 7, 'two_qubit_cost': 7, 'gate_count': 1}
        ]
    
    def create_basic_circuit(self):
        qr = QuantumRegister(9, 'q')
        cr = ClassicalRegister(8, 'c')
        circ = QuantumCircuit(qr, cr)
        return circ
    
    def test_circuit(self, circuit, input_val, expected_output):
        test_circ = circuit.copy()
        
        for i in range(8):
            if (input_val >> i) & 1:
                test_circ.x(i)
        
        for q in range(8):
            test_circ.measure(q, q)
        
        simulator = AerSimulator()
        compiled_circuit = transpile(test_circ, simulator)
        result = simulator.run(compiled_circuit, shots=1024).result()
        counts = result.get_counts()
        
        most_frequent = max(counts, key=counts.get)
        most_frequent_int = int(most_frequent, 2)
        
        return most_frequent_int == expected_output
    
    def test_circuit_full(self, circuit):
        correct_count = 0
        for i in range(256):
            if self.test_circuit(circuit, i, aes_sbox[i]):
                correct_count += 1
        return correct_count / 256.0
    
    def evaluate_circuit(self, circuit):
        correctness = self.test_circuit_full(circuit)
        gate_penalty = 0.005 * len(circuit.data)
        complexity_penalty = 0.0001 * self.quantum_cost
        score = correctness - gate_penalty - complexity_penalty
        return score, correctness
    
    def crossover(self, parent1, parent2):
        child = self.create_basic_circuit()
        
        max_gates = max(len(parent1.data), len(parent2.data))
        crossover_point = random.randint(0, max_gates)
        
        for i in range(crossover_point):
            if i < len(parent1.data):
                child.data.append(parent1.data[i])
        
        for i in range(crossover_point, max_gates):
            if i < len(parent2.data):
                child.data.append(parent2.data[i])
        
        return child
    
    def mutate(self, circuit, mutation_rate=0.3):
        gates = self.build_gate_set()
        mutated_circuit = circuit.copy()
        
        if random.random() < mutation_rate:
            gate = random.choice(gates)
            qubits = np.random.choice(8, gate['qubits'], replace=False)
            
            if gate['name'] == 'NOT':
                mutated_circuit.x(qubits[0])
            elif gate['name'] == 'CNOT':
                mutated_circuit.cx(qubits[0], qubits[1])
            elif gate['name'] == 'Toffoli':
                mutated_circuit.ccx(qubits[0], qubits[1], qubits[2])
            elif gate['name'] == 'controlled_Toffoli':
                control_qubits = qubits[:-1]
                target_qubit = qubits[-1]
                self.multi_controlled_toffoli(mutated_circuit, control_qubits, target_qubit)
        
        if random.random() < 0.2 and len(mutated_circuit.data) > 0:
            gate_idx = np.random.randint(0, len(mutated_circuit.data))
            mutated_circuit.data.pop(gate_idx)
        
        return mutated_circuit
    
    def optimize_with_genetic_algorithm(self, population_size=50, generations=100, elitism_rate=0.1, mutation_rate=0.3):
        population = []
        scores = []
        
        for _ in range(population_size):
            circ = self.create_basic_circuit()
            
            num_gates = np.random.randint(20, 100)
            gates = self.build_gate_set()
            
            for _ in range(num_gates):
                gate = np.random.choice(gates)
                qubits = np.random.choice(8, gate['qubits'], replace=False)
                
                if gate['name'] == 'NOT':
                    circ.x(qubits[0])
                elif gate['name'] == 'CNOT':
                    circ.cx(qubits[0], qubits[1])
                elif gate['name'] == 'Toffoli':
                    circ.ccx(qubits[0], qubits[1], qubits[2])
                elif gate['name'] == 'controlled_Toffoli':
                    control_qubits = qubits[:-1]
                    target_qubit = qubits[-1]
                    self.multi_controlled_toffoli(circ, control_qubits, target_qubit)
            
            population.append(circ)
            score, correctness = self.evaluate_circuit(circ)
            scores.append((score, correctness))
        
        best_overall_circuit = None
        best_overall_score = -float('inf')
        best_overall_correctness = 0
        
        for generation in range(generations):
            sorted_indices = np.argsort([-s[0] for s in scores])
            sorted_population = [population[i] for i in sorted_indices]
            sorted_scores = [scores[i] for i in sorted_indices]
            
            if sorted_scores[0][0] > best_overall_score:
                best_overall_circuit = sorted_population[0].copy()
                best_overall_score = sorted_scores[0][0]
                best_overall_correctness = sorted_scores[0][1]
                print(f"Generation {generation}: New best score {best_overall_score}, correctness {best_overall_correctness}")
            
            if best_overall_correctness == 1.0:
                print("Perfect S-box implementation achieved!")
                break
            
            next_population = []
            next_scores = []
            
            num_elite = max(1, int(population_size * elitism_rate))
            for i in range(num_elite):
                next_population.append(sorted_population[i].copy())
                next_scores.append(sorted_scores[i])
            
            while len(next_population) < population_size:
                tournament_size = 5
                tournament_indices = np.random.choice(len(sorted_population), tournament_size, replace=False)
                tournament_scores = [sorted_scores[i][0] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_scores)]
                
                tournament_indices = np.random.choice(len(sorted_population), tournament_size, replace=False)
                tournament_scores = [sorted_scores[i][0] for i in tournament_indices]
                parent2_idx = tournament_indices[np.argmax(tournament_scores)]
                
                parent1 = sorted_population[parent1_idx]
                parent2 = sorted_population[parent2_idx]
                
                child = self.crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self.mutate(child)
                
                score, correctness = self.evaluate_circuit(child)
                
                next_population.append(child)
                next_scores.append((score, correctness))
            
            population = next_population
            scores = next_scores
        
        return best_overall_circuit, best_overall_score, best_overall_correctness
    
    def implement_aes_sbox(self, optimization_method='genetic', population_size=50,generations=100, mutation_rate=0.3):
        start_time = time.time()
        
        if optimization_method == 'genetic':
            optimized_circuit, score, correctness = self.optimize_with_genetic_algorithm(
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        end_time = time.time()
        
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        print(f"Final score: {score}")
        print(f"Final correctness: {correctness}")
        print(f"Gate count: {len(optimized_circuit.data)}")
        
        test_results = {}
        for i in range(256):
            test_circ = optimized_circuit.copy()
            
            for j in range(8):
                if (i >> j) & 1:
                    test_circ.x(j)
            
            for q in range(8):
                test_circ.measure(q, q)
            
            simulator = AerSimulator()
            compiled_circuit = transpile(test_circ, simulator)
            result = simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            output = int(list(counts.keys())[0], 2)
            expected = aes_sbox[i]
            
            test_results[i] = {
                'input': i,
                'output': output,
                'expected': expected,
                'correct': output == expected
            }
        
        return optimized_circuit, test_results

def main():
    builder = AdvancedReversibleSboxBuilder(seed=42)
    print("Starting AES S-box implementation...")
    
    # Experiment with different parameters
    optimization_parameters = [
        {'population_size': 50, 'generations': 100, 'mutation_rate': 0.3},
        {'population_size': 100, 'generations': 150, 'mutation_rate': 0.4},
        {'population_size': 75, 'generations': 200, 'mutation_rate': 0.25}
    ]
    
    best_result = None
    best_correctness = 0
    
    for params in optimization_parameters:
        print("\n--- New Optimization Attempt ---")
        print(f"Parameters: {params}")
        
        try:
            circuit, results = builder.implement_aes_sbox(
                population_size=params['population_size'],
                generations=params['generations'],
                mutation_rate=params['mutation_rate']
            )
            
            correct_count = sum(1 for r in results.values() if r['correct'])
            correctness_percentage = correct_count / 256 * 100
            
            print(f"Circuit depth: {circuit.depth()}")
            print(f"Circuit size: {circuit.size()}")
            print(f"Correct outputs: {correct_count}/256 ({correctness_percentage:.2f}%)")
            
            # Track the best result
            if correctness_percentage > best_correctness:
                best_result = (circuit, results, params)
                best_correctness = correctness_percentage
        
        except Exception as e:
            print(f"Error in optimization attempt: {e}")
    
    # Final reporting of best result
    if best_result:
        circuit, results, params = best_result
        print("\n--- Best Result ---")
        print(f"Best Parameters: {params}")
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Circuit size: {circuit.size()}")
        
        correct_count = sum(1 for r in results.values() if r['correct'])
        print(f"Correct outputs: {correct_count}/256 ({correct_count/256*100:.2f}%)")
        
        # Detailed error analysis
        failed_cases = [
            r for r in results.values() if not r['correct']
        ]
        print("\nFailed test cases:")
        for case in failed_cases[:10]:  # Print first 10 failed cases
            print(f"Input: {case['input']}, Expected: {case['expected']}, Got: {case['output']}")
        
        # Optional: Visualize the circuit
        # print(circuit.draw())

if __name__ == '__main__':
    main()