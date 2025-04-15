import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import random
import csv
import json

class MixColumnsQuantumCircuitRL:
    def __init__(self, input_state, gate_config):
        self.input_state = input_state
        self.gate_config = gate_config
    
    def create_quantum_circuit(self):
        # Create quantum circuit
        qr = QuantumRegister(32, 'b')
        cr = ClassicalRegister(32, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Initialize state
        binary_state = [int(b) for b in f'{self.input_state:032b}']
        for i, bit in enumerate(reversed(binary_state)):
            if bit:
                circuit.x(qr[i])
        
        # Apply gates based on configuration
        used_qubits = set()
        for gate_info in self.gate_config:
            gate_type, q1, q2, q3 = gate_info
            
            # Ensure unique qubits
            def get_unique_qubits(num_qubits):
                available_qubits = [q for q in range(32) if q not in used_qubits]
                if len(available_qubits) < num_qubits:
                    available_qubits = list(range(32))
                
                selected_qubits = random.sample(available_qubits, num_qubits)
                used_qubits.update(selected_qubits)
                return selected_qubits
            
            if gate_type == 'cx':
                q1, q2 = get_unique_qubits(2)
                circuit.cx(qr[q1], qr[q2])
            elif gate_type == 'ccx':
                q1, q2, q3 = get_unique_qubits(3)
                circuit.ccx(qr[q1], qr[q2], qr[q3])
            elif gate_type == 'swap':
                q1, q2 = get_unique_qubits(2)
                circuit.swap(qr[q1], qr[q2])
        
        # Measure
        circuit.measure(qr, cr)
        
        return circuit

def classical_mix_columns(state):
    b0, b1, b2, b3 = (
        (state >> 24) & 0xFF, 
        (state >> 16) & 0xFF, 
        (state >> 8) & 0xFF, 
        state & 0xFF
    )
    
    new_b0 = b0 ^ b2
    new_b1 = b1 ^ b0 ^ b2
    new_b2 = b2 ^ b1
    new_b3 = b3
    
    return (new_b0 << 24) | (new_b1 << 16) | (new_b2 << 8) | new_b3

class QuantumCircuitEvolutionAgent:
    def __init__(self, population_size=30, mutation_rate=0.2, max_gates=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_gates = max_gates
        self.population = self._initialize_population()
        self.simulator = AerSimulator(method='matrix_product_state')
    
    def _generate_random_circuit_config(self):
        num_gates = random.randint(3, self.max_gates)
        circuit_config = []
        
        for _ in range(num_gates):
            gate_type = random.choice(['cx', 'ccx', 'swap'])
            
            if gate_type == 'cx':
                q1, q2 = random.sample(range(32), 2)
                circuit_config.append([gate_type, q1, q2, 0])
            elif gate_type == 'ccx':
                q1, q2, q3 = random.sample(range(32), 3)
                circuit_config.append([gate_type, q1, q2, q3])
            elif gate_type == 'swap':
                q1, q2 = random.sample(range(32), 2)
                circuit_config.append([gate_type, q1, q2, 0])
        
        return circuit_config
    
    def _initialize_population(self):
        return [self._generate_random_circuit_config() 
                for _ in range(self.population_size)]
    
    def fitness_evaluation(self, test_vectors):
        fitness_scores = []
        
        for circuit_config in self.population:
            passed_tests = 0
            total_tests = len(test_vectors)
            
            for input_state, classical_output in test_vectors:
                # Create quantum circuit
                qc_instance = MixColumnsQuantumCircuitRL(input_state, circuit_config)
                circuit = qc_instance.create_quantum_circuit()
                
                # Run simulation with fewer shots to reduce memory
                job = self.simulator.run(circuit, shots=1)
                result = job.result()
                counts = result.get_counts()
                
                # Get measured state
                measured_state = list(counts.keys())[0]
                quantum_output = int(measured_state[::-1], 2)
                
                if quantum_output == classical_output:
                    passed_tests += 1
            
            fitness = passed_tests / total_tests
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def selection(self, fitness_scores):
        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(range(len(self.population)), 5)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            selected.append(self.population[winner])
        return selected
    
    def crossover(self, selected_population):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected_population, 2)
            
            # Uniform crossover
            child = []
            for p1_gate, p2_gate in zip(parent1, parent2):
                child.append(p1_gate if random.random() < 0.5 else p2_gate)
            
            new_population.append(child)
        
        return new_population
    
    def mutation(self, population):
        mutated_population = []
        
        for circuit_config in population:
            mutated_config = [gate.copy() for gate in circuit_config]
            
            for i in range(len(mutated_config)):
                if random.random() < self.mutation_rate:
                    gate_type = random.choice(['cx', 'ccx', 'swap'])
                    
                    if gate_type == 'cx':
                        q1, q2 = random.sample(range(32), 2)
                        mutated_config[i] = [gate_type, q1, q2, 0]
                    elif gate_type == 'ccx':
                        q1, q2, q3 = random.sample(range(32), 3)
                        mutated_config[i] = [gate_type, q1, q2, q3]
                    elif gate_type == 'swap':
                        q1, q2 = random.sample(range(32), 2)
                        mutated_config[i] = [gate_type, q1, q2, 0]
            
            mutated_population.append(mutated_config)
        
        return mutated_population
    
    def evolve(self, test_vectors, generations=30):
        best_fitness = 0
        best_circuit_config = None
        
        for generation in range(generations):
            # Fitness evaluation
            fitness_scores = self.fitness_evaluation(test_vectors)
            
            # Track best circuit configuration
            current_best_fitness = max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_circuit_config = self.population[fitness_scores.index(current_best_fitness)]
            
            # Print generation progress
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")
            
            # Selection
            selected_population = self.selection(fitness_scores)
            
            # Crossover
            population = self.crossover(selected_population)
            
            # Mutation
            population = self.mutation(population)
            
            # Update population
            self.population = population
        
        return best_circuit_config, best_fitness

def generate_comprehensive_test_vectors():
    test_vectors = []
    
    bit_patterns = [0x00, 0x01, 0x10, 0x11, 0x55, 0xAA, 0x0F, 0xF0, 0xFF]
    
    # Generate combinations
    for b0 in bit_patterns:
        for b1 in bit_patterns:
            for b2 in bit_patterns:
                for b3 in bit_patterns:
                    state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
                    result = classical_mix_columns(state)
                    test_vectors.append((state, result))
    
    # Random states
    for _ in range(5000):
        state = random.getrandbits(32)
        result = classical_mix_columns(state)
        test_vectors.append((state, result))
    
    # Edge cases
    edge_cases = [
        0x00000000, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA,
        0x0F0F0F0F, 0xF0F0F0F0, 0x00FF00FF, 0xFF00FF00,
        0x12345678, 0x87654321, 0xDEADBEEF, 0xC0FFEEEE
    ]
    
    for case in edge_cases:
        result = classical_mix_columns(case)
        test_vectors.append((case, result))
    
    # Remove duplicates
    unique_test_vectors = []
    seen = set()
    for vec in test_vectors:
        if vec not in seen:
            unique_test_vectors.append(vec)
            seen.add(vec)
    
    return unique_test_vectors

def test_best_quantum_circuit(test_vectors, best_circuit_config):
    print("Quantum Mix Columns Circuit Testing:")
    print("-" * 50)
    
    total_tests = len(test_vectors)
    passed_tests = 0
    failed_tests = []
    simulator = AerSimulator(method='matrix_product_state')
    
    for input_state, classical_output in test_vectors:
        # Create quantum circuit
        qc_instance = MixColumnsQuantumCircuitRL(input_state, best_circuit_config)
        circuit = qc_instance.create_quantum_circuit()
        
        # Run simulation
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Get measured state
        measured_state = list(counts.keys())[0]
        quantum_output = int(measured_state[::-1], 2)
        
        if quantum_output == classical_output:
            passed_tests += 1
        else:
            failed_tests.append({
                'input': input_state,
                'classical_output': classical_output,
                'quantum_output': quantum_output
            })
    
    print("-" * 50)
    print(f"Test Summary:")
    print(f"Total Tests:  {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {len(failed_tests)}")
    print(f"Success Rate: {passed_tests/total_tests*100:.2f}%")
    
    if failed_tests:
        with open('failed_mix_columns_tests.csv', 'w', newline='') as csvfile:
            fieldnames = ['Input (Hex)', 'Classical Output (Hex)', 'Quantum Output (Hex)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for test in failed_tests[:100]:
                writer.writerow({
                    'Input (Hex)': f'0x{test["input"]:08X}',
                    'Classical Output (Hex)': f'0x{test["classical_output"]:08X}',
                    'Quantum Output (Hex)': f'0x{test["quantum_output"]:08X}'
                })
        print(f"\nFirst 100 failed tests saved to 'failed_mix_columns_tests.csv'")
    
    # Save best circuit configuration
    with open('best_circuit_config.json', 'w') as f:
        json.dump(best_circuit_config, f)
    
    return passed_tests, total_tests

if __name__ == "__main__":
    # Generate test vectors
    test_vectors = generate_comprehensive_test_vectors()
    
    # Initialize Evolutionary Agent
    agent = QuantumCircuitEvolutionAgent(
        population_size=30, 
        mutation_rate=0.2, 
        max_gates=10
    )
    
    # Evolve the best quantum circuit configuration
    best_circuit_config, best_fitness = agent.evolve(test_vectors, generations=30)
    
    # Test the best quantum circuit
    passed_tests, total_tests = test_best_quantum_circuit(test_vectors, best_circuit_config)