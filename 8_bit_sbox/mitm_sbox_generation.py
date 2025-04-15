from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import numpy as np
import time
import random
import itertools

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

class HybridReversibleSboxBuilder:
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
    
    def generate_half_circuit_population(self, population_size=50, max_gates=50):
        first_half_qubits, second_half_qubits = self.create_circuit_halves()
        population = []
        
        for _ in range(population_size):
            first_half_circuit = QuantumCircuit(len(first_half_qubits), len(first_half_qubits))
            second_half_circuit = QuantumCircuit(len(second_half_qubits), len(second_half_qubits))
            
            num_gates_first = random.randint(1, max_gates)
            gates = self.build_gate_set()
            
            for _ in range(num_gates_first):
                gate = random.choice(gates)
                
                if gate['name'] == 'NOT':
                    qubit = random.choice(range(len(first_half_qubits)))
                    first_half_circuit.x(qubit)
                elif gate['name'] == 'CNOT':
                    qubits = random.sample(range(len(first_half_qubits)), 2)
                    first_half_circuit.cx(qubits[0], qubits[1])
                elif gate['name'] == 'Toffoli':
                    qubits = random.sample(range(len(first_half_qubits)), 3)
                    first_half_circuit.ccx(qubits[0], qubits[1], qubits[2])
            
            num_gates_second = random.randint(1, max_gates)
            for _ in range(num_gates_second):
                gate = random.choice(gates)
                
                if gate['name'] == 'NOT':
                    qubit = random.choice(range(len(second_half_qubits)))
                    second_half_circuit.x(qubit)
                elif gate['name'] == 'CNOT':
                    qubits = random.sample(range(len(second_half_qubits)), 2)
                    second_half_circuit.cx(qubits[0], qubits[1])
                elif gate['name'] == 'Toffoli':
                    qubits = random.sample(range(len(second_half_qubits)), 3)
                    second_half_circuit.ccx(qubits[0], qubits[1], qubits[2])
            
            population.append((first_half_circuit, second_half_circuit))
        
        return population
    
    def create_circuit_halves(self, total_qubits=8):
        mid_point = total_qubits // 2
        first_half_qubits = list(range(mid_point))
        second_half_qubits = list(range(mid_point, total_qubits))
        
        return first_half_qubits, second_half_qubits
    
    def evaluate_half_circuit_pair(self, first_half, second_half):
        first_half_inputs = list(itertools.product([0, 1], repeat=len(first_half.qubits)))
        second_half_inputs = list(itertools.product([0, 1], repeat=len(second_half.qubits)))
        
        first_half_mapping = {}
        second_half_mapping = {}
        
        for input_pattern in first_half_inputs:
            test_circ = first_half.copy()
            for j, val in enumerate(input_pattern):
                if val:
                    test_circ.x(j)
            
            for q in range(len(first_half.qubits)):
                test_circ.measure(q, q)
            
            compiled_circuit = transpile(test_circ, self.simulator)
            result = self.simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            output_val = int(list(counts.keys())[0], 2)
            first_half_mapping[input_pattern] = output_val
        
        for input_pattern in second_half_inputs:
            test_circ = second_half.copy()
            for j, val in enumerate(input_pattern):
                if val:
                    test_circ.x(j)
            
            for q in range(len(second_half.qubits)):
                test_circ.measure(q, q)
            
            compiled_circuit = transpile(test_circ, self.simulator)
            result = self.simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            output_val = int(list(counts.keys())[0], 2)
            second_half_mapping[input_pattern] = output_val
        
        correct_combinations = 0
        total_combinations = 256
        
        for input_val in range(total_combinations):
            first_input = input_val & ((1 << 4) - 1)
            second_input = input_val >> 4
            
            first_half_output = first_half_mapping[tuple([int(bool(first_input & (1 << j))) for j in range(4)])]
            second_half_output = second_half_mapping[tuple([int(bool(second_input & (1 << j))) for j in range(4)])]
            
            combined_output = (first_half_output << 4) | second_half_output
            
            if combined_output == aes_sbox[input_val]:
                correct_combinations += 1
        
        return correct_combinations / total_combinations
    
    def meet_in_the_middle_genetic_optimization(self, population_size=50, generations=100, elitism_rate=0.1, mutation_rate=0.3):
        start_time = time.time()
        
        population = self.generate_half_circuit_population(population_size)
        
        population_scores = []
        for first_half, second_half in population:
            score = self.evaluate_half_circuit_pair(first_half, second_half)
            population_scores.append(score)
        
        best_overall_circuit = None
        best_overall_score = -float('inf')
        
        for generation in range(generations):
            sorted_indices = np.argsort([-s for s in population_scores])
            sorted_population = [population[i] for i in sorted_indices]
            sorted_scores = [population_scores[i] for i in sorted_indices]
            
            if sorted_scores[0] > best_overall_score:
                best_overall_score = sorted_scores[0]
                best_overall_circuit = sorted_population[0]
                print(f"Generation {generation}: New best score {best_overall_score}")
            
            if best_overall_score == 1.0:
                print("Perfect S-box implementation achieved!")
                break
            
            next_population = []
            next_scores = []
            
            num_elite = max(1, int(population_size * elitism_rate))
            for i in range(num_elite):
                next_population.append(sorted_population[i])
                next_scores.append(sorted_scores[i])
            
            while len(next_population) < population_size:
                tournament_indices = np.random.choice(len(sorted_population), 5, replace=False)
                tournament_scores = [sorted_scores[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_scores)]
                
                tournament_indices = np.random.choice(len(sorted_population), 5, replace=False)
                tournament_scores = [sorted_scores[i] for i in tournament_indices]
                parent2_idx= tournament_indices[np.argmax(tournament_scores)]
                
                parent1_first, parent1_second = sorted_population[parent1_idx]
                parent2_first, parent2_second = sorted_population[parent2_idx]
                
                child_first = self.crossover(parent1_first, parent2_first)
                child_second = self.crossover(parent1_second, parent2_second)
                
                if random.random() < mutation_rate:
                    child_first = self.mutate(child_first)
                if random.random() < mutation_rate:
                    child_second = self.mutate(child_second)
                
                child_score = self.evaluate_half_circuit_pair(child_first, child_second)
                
                next_population.append((child_first, child_second))
                next_scores.append(child_score)
            
            population = next_population
            population_scores = next_scores
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        return best_overall_circuit
    
    def crossover(self, parent1, parent2):
        child = parent1.copy()
        
        max_gates = max(len(parent1.data), len(parent2.data))
        crossover_point = random.randint(0, max_gates)
        
        child.data = []
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
            qubits = np.random.choice(len(circuit.qubits), gate['qubits'], replace=False)
            
            if gate['name'] == 'NOT':
                mutated_circuit.x(qubits[0])
            elif gate['name'] == 'CNOT':
                mutated_circuit.cx(qubits[0], qubits[1])
            elif gate['name'] == 'Toffoli':
                mutated_circuit.ccx(qubits[0], qubits[1], qubits[2])
        
        if random.random() < 0.2 and len(mutated_circuit.data) > 0:
            gate_idx = np.random.randint(0, len(mutated_circuit.data))
            mutated_circuit.data.pop(gate_idx)
        
        return mutated_circuit
    
    def implement_aes_sbox(self, population_size=50, generations=100, mutation_rate=0.3):
        start_time = time.time()
        
        optimized_circuit_pair = self.meet_in_the_middle_genetic_optimization(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        
        full_circuit = QuantumCircuit(8, 8)
        first_half, second_half = optimized_circuit_pair
        
        for gate in first_half.data:
            full_circuit.data.append(gate)
        for gate in second_half.data:
            full_circuit.data.append(gate)
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        test_results = {}
        for i in range(256):
            test_circ = full_circuit.copy()
            
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
        
        return full_circuit, test_results

def main():
    builder = HybridReversibleSboxBuilder(seed=42)
    print("Starting AES S-box implementation with Hybrid Optimization...")
    
    circuit, results = builder.implement_aes_sbox()
    
    correct_count = sum(1 for r in results.values() if r['correct'])
    correctness_percentage = correct_count / 256 * 100
    
    print(f"\n--- Optimization Results ---")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Circuit size: {circuit.size()}")
    print(f"Correct outputs: {correct_count}/256 ({correctness_percentage:.2f}%)")
    
    failed_cases = [r for r in results.values() if not r['correct']]
    print("\nFailed test cases:")
    for case in failed_cases[:10]:
        print(f"Input: {case['input']}, Expected: {case['expected']}, Got: {case['output']}")

if __name__ == '__main__':
    main()