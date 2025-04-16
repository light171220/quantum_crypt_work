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

class MITMQuantumSboxOptimizer:
    def __init__(self, seed=42, num_qubits=8):
        np.random.seed(seed)
        random.seed(seed)
        self.simulator = AerSimulator()
        self.num_qubits = num_qubits
        
        self.mitm_config = {
            'first_half': list(range(num_qubits // 2)),
            'second_half': list(range(num_qubits // 2, num_qubits))
        }
    
    def create_half_circuit(self, half_qubits):
        return QuantumCircuit(len(half_qubits), len(half_qubits))
    
    def advanced_half_circuit_generation(self, circuit, generation_stage):
        gate_types = [
            lambda c, q: c.x(q),
            lambda c, q1, q2: c.cx(q1, q2),
            lambda c, q1, q2, q3: c.ccx(q1, q2, q3)
        ]
        
        max_gates = self._calculate_dynamic_gate_limit(generation_stage)
        gate_count = random.randint(1, max_gates)
        
        for _ in range(gate_count):
            gate_choice_probability = self._get_contextual_gate_probabilities(circuit)
            gate = np.random.choice(gate_types, p=gate_choice_probability)
            
            if gate == gate_types[0]:  # x gate
                qubit = random.randint(0, circuit.num_qubits - 1)
                gate(circuit, qubit)
            elif gate == gate_types[1]:  # cx gate
                qubits = random.sample(range(circuit.num_qubits), 2)
                gate(circuit, qubits[0], qubits[1])
            else:  # ccx gate
                qubits = random.sample(range(circuit.num_qubits), 3)
                gate(circuit, qubits[0], qubits[1], qubits[2])
        
        return circuit
    
    def _calculate_dynamic_gate_limit(self, generation_stage):
        base_max_gates = 10
        return int(base_max_gates * (1 - 0.1 * generation_stage))
    
    def _get_contextual_gate_probabilities(self, circuit):
        current_gate_distribution = self._analyze_current_gate_distribution(circuit)
        
        base_probabilities = [0.4, 0.4, 0.2]  # x, cx, ccx
        
        adaptive_probabilities = [
            base_probabilities[i] * (1 + self._calculate_contextual_bias(current_gate_distribution, i))
            for i in range(len(base_probabilities))
        ]
        
        normalized_probs = [p/sum(adaptive_probabilities) for p in adaptive_probabilities]
        return normalized_probs
    
    def _analyze_current_gate_distribution(self, circuit):
        gate_counts = {'x': 0, 'cx': 0, 'ccx': 0}
        for instruction, _, _ in circuit.data:
            if instruction.name == 'x':
                gate_counts['x'] += 1
            elif instruction.name == 'cx':
                gate_counts['cx'] += 1
            elif instruction.name == 'ccx':
                gate_counts['ccx'] += 1
        
        return gate_counts
    
    def _calculate_contextual_bias(self, gate_distribution, gate_index):
        total_gates = sum(gate_distribution.values())
        current_type_ratio = gate_distribution[list(gate_distribution.keys())[gate_index]] / total_gates if total_gates > 0 else 0
        return 0.5 - current_type_ratio
    
    def mitm_circuit_crossover(self, first_half1, second_half1, first_half2, second_half2):
        merged_first_half = self._probabilistic_circuit_merge(first_half1, first_half2)
        merged_second_half = self._probabilistic_circuit_merge(second_half1, second_half2)
        
        return merged_first_half, merged_second_half
    
    def _probabilistic_circuit_merge(self, circuit1, circuit2):
        merged_circuit = self.create_half_circuit(list(range(circuit1.num_qubits)))
        
        for instruction in circuit1.data + circuit2.data:
            if random.random() < 0.5:
                merged_circuit.append(instruction[0], instruction[1])
        
        return merged_circuit
    
    def adaptive_half_circuit_mutation(self, circuit, mutation_rate, generation_stage):
        dynamic_mutation_rate = mutation_rate * (1 - 0.1 * generation_stage)
        mutated = circuit.copy()
        
        if random.random() < dynamic_mutation_rate:
            gate_types = [
                lambda c, q: c.x(q),
                lambda c, q1, q2: c.cx(q1, q2),
                lambda c, q1, q2, q3: c.ccx(q1, q2, q3)
            ]
            
            mutation_strategy = random.choice([
                self._local_mutation,
                self._global_mutation,
                self._reversibility_preserving_mutation
            ])
            
            mutated = mutation_strategy(mutated, gate_types)
        
        return mutated
    
    def _local_mutation(self, circuit, gate_types):
        gate = random.choice(gate_types)
        
        if gate == gate_types[0]:
            qubit = random.randint(0, circuit.num_qubits - 1)
            gate(circuit, qubit)
        elif gate == gate_types[1]:
            qubits = random.sample(range(circuit.num_qubits), 2)
            gate(circuit, qubits[0], qubits[1])
        else:
            qubits = random.sample(range(circuit.num_qubits), 3)
            gate(circuit, qubits[0], qubits[1], qubits[2])
        
        return circuit
    
    def _global_mutation(self, circuit, gate_types):
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            self._local_mutation(circuit, gate_types)
        return circuit
    
    def _reversibility_preserving_mutation(self, circuit, gate_types):
        mutated = circuit.copy()
        two_qubit_gates = sum(1 for inst in mutated.data if inst[0].name == 'cx')
        three_qubit_gates = sum(1 for inst in mutated.data if inst[0].name == 'ccx')
        
        max_two_qubit_gates = 5
        max_three_qubit_gates = 3
        
        if two_qubit_gates < max_two_qubit_gates and three_qubit_gates < max_three_qubit_gates:
            gate = random.choice([gate_types[1], gate_types[2]])
            qubits = random.sample(range(mutated.num_qubits), 3 if gate == gate_types[2] else 2)
            
            if gate == gate_types[1]:
                gate(mutated, qubits[0], qubits[1])
            else:
                gate(mutated, qubits[0], qubits[1], qubits[2])
        
        return mutated
    
    def evaluate_mitm_circuit_pair(self, first_half, second_half):
        correct_combinations = 0
        total_combinations = 256
        
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
            
            output = int(list(counts.keys())[0], 2)
            first_half_mapping[input_pattern] = output
        
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
            
            output = int(list(counts.keys())[0], 2)
            second_half_mapping[input_pattern] = output
        
        for input_val in range(total_combinations):
            first_input = input_val & ((1 << (self.num_qubits // 2)) - 1)
            second_input = input_val >> (self.num_qubits // 2)
            
            first_half_output = first_half_mapping[tuple([int(bool(first_input & (1 << j))) for j in range(len(first_half.qubits))])]
            second_half_output = second_half_mapping[tuple([int(bool(second_input & (1 << j))) for j in range(len(second_half.qubits))])]
            
            combined_output = (first_half_output << len(second_half.qubits)) | second_half_output
            
            if combined_output == aes_sbox[input_val]:
                correct_combinations += 1
        
        return correct_combinations / total_combinations
    
    def mitm_genetic_optimization(self, population_size=50, generations=100, mutation_rate=0.3):
        start_time = time.time()
        
        first_half_population = [
            self.create_half_circuit(self.mitm_config['first_half']) for _ in range(population_size)
        ]
        second_half_population = [
            self.create_half_circuit(self.mitm_config['second_half']) for _ in range(population_size)
        ]
        
        first_half_scores = []
        second_half_scores = []
        
        for gen in range(generations):
            for i in range(population_size):
                self.advanced_half_circuit_generation(first_half_population[i], gen)
                self.advanced_half_circuit_generation(second_half_population[i], gen)
            
            first_half_scores = [
                self.evaluate_mitm_circuit_pair(first_half_population[i], random.choice(second_half_population))
                for i in range(population_size)
            ]
            
            second_half_scores = [
                self.evaluate_mitm_circuit_pair(random.choice(first_half_population), second_half_population[i])
                for i in range(population_size)
            ]
            
            print(f"Generation {gen}: Best First Half Score = {max(first_half_scores)}, Best Second Half Score = {max(second_half_scores)}")
            
            if max(first_half_scores) == 1.0 and max(second_half_scores) == 1.0:
                break
            
            first_half_sorted_indices = np.argsort(first_half_scores)[::-1]
            second_half_sorted_indices = np.argsort(second_half_scores)[::-1]
            
            first_half_population = [first_half_population[i] for i in first_half_sorted_indices]
            second_half_population = [second_half_population[i] for i in second_half_sorted_indices]
            
            next_first_half_population = first_half_population[:population_size//2]
            next_second_half_population = second_half_population[:population_size//2]
            
            while len(next_first_half_population) < population_size:
                parent1_idx, parent2_idx = random.sample(range(population_size//2), 2)
                
                child_first_half1, child_second_half1 = next_first_half_population[parent1_idx], second_half_population[parent1_idx]
                child_first_half2, child_second_half2 = next_first_half_population[parent2_idx], second_half_population[parent2_idx]
                
                merged_first_half, merged_second_half = self.mitm_circuit_crossover(
                    child_first_half1, child_second_half1, 
                    child_first_half2, child_second_half2
                )
                
                merged_first_half = self.adaptive_half_circuit_mutation(merged_first_half, mutation_rate, gen)
                merged_second_half = self.adaptive_half_circuit_mutation(merged_second_half, mutation_rate, gen)
                
                next_first_half_population.append(merged_first_half)
                next_second_half_population.append(merged_second_half)
            
            first_half_population = next_first_half_population
            second_half_population = next_second_half_population
        
        end_time = time.time()
        print(f"MITM Optimization completed in {end_time - start_time:.2f} seconds")
        
        best_first_half_idx = np.argmax(first_half_scores)
        best_second_half_idx = np.argmax(second_half_scores)
        
        return first_half_population[best_first_half_idx], second_half_population[best_second_half_idx]
    
    def combine_mitm_circuits(self, first_half, second_half):
        full_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        for gate in first_half.data:
            full_circuit.append(gate[0], gate[1])
        
        for gate in second_half.data:
            full_circuit.append(gate[0], gate[1])
        
        return full_circuit
    
    def test_final_mitm_circuit(self, circuit):
        results = {}
        for i in range(256):
            test_circuit = circuit.copy()
            
            for j in range(self.num_qubits):
                if (i >> j) & 1:
                    test_circuit.x(j)
            
            for q in range(self.num_qubits):
                test_circuit.measure(q, q)
            
            compiled_circuit = transpile(test_circuit, self.simulator)
            result = self.simulator.run(compiled_circuit, shots=1).result()
            counts = result.get_counts()
            
            output = int(list(counts.keys())[0], 2)
            
            results[i] = {
                'input': i,
                'output': output,
                'expected': aes_sbox[i],
                'correct': output == aes_sbox[i]
            }
        
        return results

def main():
    start_time = time.time()
    
    optimizer = MITMQuantumSboxOptimizer(seed=42)
    print("Starting Meet-in-the-Middle Quantum S-box Optimization...")
    
    first_half_circuit, second_half_circuit = optimizer.mitm_genetic_optimization()
    
    full_circuit = optimizer.combine_mitm_circuits(first_half_circuit, second_half_circuit)
    
    print("\nOptimization Complete. Testing Circuit...")
    
    results = optimizer.test_final_mitm_circuit(full_circuit)
    
    correct_count = sum(1 for r in results.values() if r['correct'])
    correctness_percentage = correct_count / 256 * 100
    
    end_time = time.time()
    
    print(f"\n--- Optimization Results ---")
    print(f"Total Optimization Time: {end_time - start_time:.2f} seconds")
    print(f"Circuit Depth: {full_circuit.depth()}")
    print(f"Circuit Size: {len(full_circuit.data)}")
    print(f"Correct Outputs: {correct_count}/256 ({correctness_percentage:.2f}%)")
    
    failed_cases = [r for r in results.values() if not r['correct']]
    print("\nFailed Test Cases:")
    for case in failed_cases[:10]:
        print(f"Input: {case['input']}, Expected: {case['expected']}, Got: {case['output']}")
    
    print("\nCircuit Visualization:")
    print(full_circuit)

if __name__ == '__main__':
    main()