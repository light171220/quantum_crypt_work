from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import numpy as np
import random
import time
import copy
import pickle
import multiprocessing as mp
from functools import partial

def high_diffusion_mix_columns(circuit, qubits):
    for col in range(4):
        col_start = col * 2
        
        c0 = col_start
        c8 = 8 + col_start
        c16 = 16 + col_start
        c24 = 24 + col_start
        
        # Phase 1: Vertical connections
        circuit.cx(qubits[c0], qubits[c8])
        circuit.cx(qubits[c0+1], qubits[c8+1])
        circuit.cx(qubits[c8], qubits[c16])
        circuit.cx(qubits[c8+1], qubits[c16+1])
        circuit.cx(qubits[c16], qubits[c24])
        circuit.cx(qubits[c16+1], qubits[c24+1])
        
        # Phase 2: Bit pair mixing within rows
        circuit.cx(qubits[c0], qubits[c0+1])
        circuit.cx(qubits[c8], qubits[c8+1])
        circuit.cx(qubits[c16], qubits[c16+1])
        circuit.cx(qubits[c24], qubits[c24+1])
        
        # Phase 3: Diagonal connections
        circuit.cx(qubits[c0], qubits[c16])
        circuit.cx(qubits[c0+1], qubits[c16+1])
        circuit.cx(qubits[c8], qubits[c24])
        circuit.cx(qubits[c8+1], qubits[c24+1])
        
        # Phase 4: Feedback from bottom to top
        circuit.cx(qubits[c24], qubits[c0])
        circuit.cx(qubits[c24+1], qubits[c0+1])
        circuit.cx(qubits[c16], qubits[c8])
        circuit.cx(qubits[c16+1], qubits[c8+1])
        
        # Phase 5: Cross connections between adjacent rows
        circuit.cx(qubits[c0], qubits[c8+1])
        circuit.cx(qubits[c8], qubits[c16+1])
        circuit.cx(qubits[c16], qubits[c24+1])
        circuit.cx(qubits[c24], qubits[c0+1])
        
        # Phase 6: Final intra-row mixing
        circuit.swap(qubits[c0], qubits[c0+1])
        circuit.swap(qubits[c16], qubits[c16+1])
    
    return circuit

def inverse_high_diffusion_mix_columns(circuit, qubits):
    for col in range(3, -1, -1):
        col_start = col * 2
        
        c0 = col_start
        c8 = 8 + col_start
        c16 = 16 + col_start
        c24 = 24 + col_start
        
        # Undo Phase 6
        circuit.swap(qubits[c16], qubits[c16+1])
        circuit.swap(qubits[c0], qubits[c0+1])
        
        # Undo Phase 5
        circuit.cx(qubits[c24], qubits[c0+1])
        circuit.cx(qubits[c16], qubits[c24+1])
        circuit.cx(qubits[c8], qubits[c16+1])
        circuit.cx(qubits[c0], qubits[c8+1])
        
        # Undo Phase 4
        circuit.cx(qubits[c16+1], qubits[c8+1])
        circuit.cx(qubits[c16], qubits[c8])
        circuit.cx(qubits[c24+1], qubits[c0+1])
        circuit.cx(qubits[c24], qubits[c0])
        
        # Undo Phase 3
        circuit.cx(qubits[c8+1], qubits[c24+1])
        circuit.cx(qubits[c8], qubits[c24])
        circuit.cx(qubits[c0+1], qubits[c16+1])
        circuit.cx(qubits[c0], qubits[c16])
        
        # Undo Phase 2
        circuit.cx(qubits[c24], qubits[c24+1])
        circuit.cx(qubits[c16], qubits[c16+1])
        circuit.cx(qubits[c8], qubits[c8+1])
        circuit.cx(qubits[c0], qubits[c0+1])
        
        # Undo Phase 1
        circuit.cx(qubits[c16], qubits[c24])
        circuit.cx(qubits[c16+1], qubits[c24+1])
        circuit.cx(qubits[c8], qubits[c16])
        circuit.cx(qubits[c8+1], qubits[c16+1])
        circuit.cx(qubits[c0], qubits[c8])
        circuit.cx(qubits[c0+1], qubits[c8+1])
    
    return circuit

def high_diffusion_to_gate_sequence():
    gate_sequence = []
    
    for col in range(4):
        col_start = col * 2
        
        c0 = col_start
        c8 = 8 + col_start
        c16 = 16 + col_start
        c24 = 24 + col_start
        
        # Phase 1: Vertical connections
        gate_sequence.append(('cx', c0, c8))
        gate_sequence.append(('cx', c0+1, c8+1))
        gate_sequence.append(('cx', c8, c16))
        gate_sequence.append(('cx', c8+1, c16+1))
        gate_sequence.append(('cx', c16, c24))
        gate_sequence.append(('cx', c16+1, c24+1))
        
        # Phase 2: Bit pair mixing within rows
        gate_sequence.append(('cx', c0, c0+1))
        gate_sequence.append(('cx', c8, c8+1))
        gate_sequence.append(('cx', c16, c16+1))
        gate_sequence.append(('cx', c24, c24+1))
        
        # Phase 3: Diagonal connections
        gate_sequence.append(('cx', c0, c16))
        gate_sequence.append(('cx', c0+1, c16+1))
        gate_sequence.append(('cx', c8, c24))
        gate_sequence.append(('cx', c8+1, c24+1))
        
        # Phase 4: Feedback from bottom to top
        gate_sequence.append(('cx', c24, c0))
        gate_sequence.append(('cx', c24+1, c0+1))
        gate_sequence.append(('cx', c16, c8))
        gate_sequence.append(('cx', c16+1, c8+1))
        
        # Phase 5: Cross connections between adjacent rows
        gate_sequence.append(('cx', c0, c8+1))
        gate_sequence.append(('cx', c8, c16+1))
        gate_sequence.append(('cx', c16, c24+1))
        gate_sequence.append(('cx', c24, c0+1))
        
        # Phase 6: Final intra-row mixing
        gate_sequence.append(('swap', c0, c0+1))
        gate_sequence.append(('swap', c16, c16+1))
    
    return gate_sequence

def apply_gate(circuit, qubits, gate_type, targets):
    if gate_type == 'cx':
        circuit.cx(qubits[targets[0]], qubits[targets[1]])
    elif gate_type == 'swap':
        circuit.swap(qubits[targets[0]], qubits[targets[1]])
    elif gate_type == 'x':
        circuit.x(qubits[targets[0]])
    return circuit

def apply_gates(circuit, qubits, gate_sequence):
    for gate in gate_sequence:
        gate_type = gate[0]
        targets = gate[1:]
        circuit = apply_gate(circuit, qubits, gate_type, targets)
    return circuit

def generate_inverse(gate_sequence):
    inverse_sequence = []
    for gate in reversed(gate_sequence):
        gate_type = gate[0]
        targets = gate[1:]
        inverse_sequence.append((gate_type, *targets))
    return inverse_sequence

def evaluate_circuit(gate_sequence, test_vectors=None, simulator=None, min_gates=10):
    if simulator is None:
        simulator = AerSimulator()
    
    if test_vectors is None:
        test_vectors = ['0' * 32, '1' * 32]
        for i in range(32):
            vector = ['0'] * 32
            vector[i] = '1'
            test_vectors.append(''.join(vector))
    
    # Penalize sequences with too few gates
    if len(gate_sequence) < min_gates:
        return {
            'success_rate': 0.0,
            'diffusion_rate': 0.0,
            'avg_bit_flips': 0.0,
            'score': 0.0
        }
    
    mix_column_success = 0
    total_bit_flips = 0
    inverse_sequence = generate_inverse(gate_sequence)
    
    for test_vector in test_vectors:
        data_register = QuantumRegister(32, 'data')
        measurement_register = ClassicalRegister(32, 'measure')
        
        forward_circuit = QuantumCircuit(data_register, measurement_register)
        
        for i in range(32):
            if test_vector[i] == '1':
                forward_circuit.x(data_register[i])
        
        forward_circuit = apply_gates(forward_circuit, data_register, gate_sequence)
        forward_circuit.measure(data_register, measurement_register)
        
        forward_result = simulator.run(forward_circuit).result()
        forward_counts = forward_result.get_counts()
        output_vector = max(forward_counts, key=forward_counts.get)
        
        # Count bit changes
        bit_flips = sum(1 for a, b in zip(test_vector, output_vector) if a != b)
        total_bit_flips += bit_flips
        
        # Check that output is different from input (to prevent identity transformation)
        if bit_flips == 0:
            # Identity transformation is bad for mix columns
            continue
        
        inverse_circuit = QuantumCircuit(data_register, measurement_register)
        
        for i in range(32):
            if output_vector[i] == '1':
                inverse_circuit.x(data_register[i])
        
        inverse_circuit = apply_gates(inverse_circuit, data_register, inverse_sequence)
        inverse_circuit.measure(data_register, measurement_register)
        
        inverse_result = simulator.run(inverse_circuit).result()
        inverse_counts = inverse_result.get_counts()
        recovered_vector = max(inverse_counts, key=inverse_counts.get)
        
        if recovered_vector == test_vector:
            mix_column_success += 1
    
    if len(test_vectors) > 0:
        success_rate = mix_column_success / len(test_vectors) * 100
        avg_bit_flips = total_bit_flips / len(test_vectors)
        diffusion_rate = avg_bit_flips / 32 * 100
    else:
        success_rate = 0
        avg_bit_flips = 0
        diffusion_rate = 0
    
    # Add gate efficiency component to the score
    gate_efficiency = max(0, min(1, 1.0 - (len(gate_sequence) - min_gates) / 100))
    
    # Calculate final score - prioritize correctness, then diffusion, with a small bonus for efficiency
    score = (success_rate * 0.7) + (diffusion_rate * 0.25) + (gate_efficiency * 5)
    
    # Ensure empty sequences or identity transformations get zero score
    if len(gate_sequence) == 0 or diffusion_rate < 5.0:
        score = 0
    
    return {
        'success_rate': success_rate,
        'diffusion_rate': diffusion_rate,
        'avg_bit_flips': avg_bit_flips,
        'score': score
    }

def generate_random_gate():
    gate_types = ['cx', 'swap']  # Removed 'x' as it's not typically used in mix columns
    gate_type = random.choice(gate_types)
    
    if gate_type == 'cx':
        control = random.randint(0, 31)
        target = random.randint(0, 31)
        while target == control:
            target = random.randint(0, 31)
        return (gate_type, control, target)
    elif gate_type == 'swap':
        qubit1 = random.randint(0, 31)
        qubit2 = random.randint(0, 31)
        while qubit2 == qubit1:
            qubit2 = random.randint(0, 31)
        return (gate_type, qubit1, qubit2)

def generate_initial_population(pop_size, base_sequence):
    population = [base_sequence]
    
    for _ in range(pop_size - 1):
        num_mutations = random.randint(1, 10)
        sequence_copy = copy.deepcopy(base_sequence)
        
        for _ in range(num_mutations):
            if random.random() < 0.6:
                # Replace a gate
                if len(sequence_copy) > 0:
                    idx = random.randint(0, len(sequence_copy) - 1)
                    sequence_copy[idx] = generate_random_gate()
            elif random.random() < 0.8:
                # Add a gate
                idx = random.randint(0, len(sequence_copy))
                sequence_copy.insert(idx, generate_random_gate())
            else:
                # Remove a gate
                if len(sequence_copy) > 20:  # Ensure we don't go below minimal gate count
                    idx = random.randint(0, len(sequence_copy) - 1)
                    sequence_copy.pop(idx)
        
        population.append(sequence_copy)
    
    return population

def mutate_gate_sequence(gate_sequence, mutation_rate, min_gates=10):
    new_sequence = copy.deepcopy(gate_sequence)
    
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            operation = random.choice(['replace', 'add', 'remove'])
            
            if operation == 'replace' or len(new_sequence) <= min_gates + 2:
                new_sequence[i] = generate_random_gate()
            elif operation == 'add':
                new_sequence.insert(i, generate_random_gate())
            elif operation == 'remove' and len(new_sequence) > min_gates + 2:
                new_sequence.pop(i)
                break
    
    if random.random() < mutation_rate:
        if len(new_sequence) > min_gates + 5 and random.random() < 0.5:
            idx = random.randint(0, len(new_sequence) - 1)
            new_sequence.pop(idx)
        else:
            new_sequence.append(generate_random_gate())
    
    return new_sequence

def crossover(parent1, parent2, min_gates=10):
    # Ensure children have enough gates
    if len(parent1) < min_gates or len(parent2) < min_gates:
        if len(parent1) >= min_gates:
            return parent1
        elif len(parent2) >= min_gates:
            return parent2
        else:
            # If both parents are too small, use the larger one and add random gates
            larger_parent = parent1 if len(parent1) > len(parent2) else parent2
            result = copy.deepcopy(larger_parent)
            while len(result) < min_gates:
                result.append(generate_random_gate())
            return result
    
    crossover_point1 = random.randint(0, len(parent1))
    crossover_point2 = random.randint(0, len(parent2))
    
    child1 = parent1[:crossover_point1] + parent2[crossover_point2:]
    child2 = parent2[:crossover_point2] + parent1[crossover_point1:]
    
    # Ensure children have enough gates
    child = child1 if random.random() < 0.5 else child2
    if len(child) < min_gates:
        while len(child) < min_gates:
            child.append(generate_random_gate())
    
    return child

def evaluate_individual(gate_sequence, test_vectors, simulator, min_gates):
    try:
        return evaluate_circuit(gate_sequence, test_vectors, simulator, min_gates)
    except Exception as e:
        print(f"Error evaluating circuit: {e}")
        return {'score': 0, 'success_rate': 0, 'diffusion_rate': 0, 'avg_bit_flips': 0}

def evaluate_population(population, test_vectors, pool, min_gates):
    simulator = AerSimulator()
    eval_func = partial(evaluate_individual, test_vectors=test_vectors, simulator=simulator, min_gates=min_gates)
    results = pool.map(eval_func, population)
    return results

def select_parents(population, fitness_scores, tournament_size):
    selected = []
    for _ in range(2):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i]['score'] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected.append(population[winner_idx])
    return selected[0], selected[1]

def genetic_algorithm(base_sequence, pop_size=40, generations=50, mutation_rate=0.3, tournament_size=3, num_processors=4, min_gates=10):
    test_vectors = ['0' * 32, '1' * 32]
    for i in range(32):
        vector = ['0'] * 32
        vector[i] = '1'
        test_vectors.append(''.join(vector))
    
    # Add more complex test vectors
    for _ in range(16):
        test_vectors.append(''.join(random.choice(['0', '1']) for _ in range(32)))
    
    population = generate_initial_population(pop_size, base_sequence)
    best_individual = base_sequence
    best_fitness = {'score': 0}
    
    with mp.Pool(processes=num_processors) as pool:
        for generation in range(generations):
            start_time = time.time()
            
            fitness_scores = evaluate_population(population, test_vectors, pool, min_gates)
            
            # Find best individual
            for i, fitness in enumerate(fitness_scores):
                if fitness['score'] > best_fitness['score']:
                    best_fitness = fitness
                    best_individual = population[i]
            
            # Status update
            elapsed = time.time() - start_time
            print(f"Generation {generation+1}/{generations} - "
                  f"Best: {best_fitness['score']:.2f} "
                  f"(Success: {best_fitness['success_rate']:.2f}%, "
                  f"Diffusion: {best_fitness['diffusion_rate']:.2f}%) "
                  f"Gates: {len(best_individual)} "
                  f"Time: {elapsed:.2f}s")
            
            # Save the best individual periodically
            if (generation + 1) % 10 == 0:
                with open(f'best_mix_column_gen{generation+1}.pkl', 'wb') as f:
                    pickle.dump({
                        'gates': best_individual,
                        'fitness': best_fitness,
                        'generation': generation + 1
                    }, f)
            
            # Early stopping condition
            if best_fitness['success_rate'] >= 99 and best_fitness['diffusion_rate'] >= 35:
                print("Found optimal solution! Stopping early.")
                break
            
            # Create the next generation
            new_population = []
            
            # Elitism - keep the best individuals
            sorted_indices = sorted(range(len(fitness_scores)), 
                                  key=lambda i: fitness_scores[i]['score'], reverse=True)
            elites = [population[i] for i in sorted_indices[:max(1, pop_size // 10)]]
            new_population.extend(elites)
            
            # Fill the rest of the population with offspring
            while len(new_population) < pop_size:
                parent1, parent2 = select_parents(population, fitness_scores, tournament_size)
                child = crossover(parent1, parent2, min_gates)
                child = mutate_gate_sequence(child, mutation_rate, min_gates)
                new_population.append(child)
            
            population = new_population
    
    # Final evaluation on a larger test set
    print("Performing final evaluation on the best individual...")
    test_vectors_extended = test_vectors.copy()
    for _ in range(50):
        test_vectors_extended.append(''.join(random.choice(['0', '1']) for _ in range(32)))
    
    final_fitness = evaluate_circuit(best_individual, test_vectors_extended, min_gates=min_gates)
    
    print(f"Final Evaluation Results:")
    print(f"Success Rate: {final_fitness['success_rate']:.2f}%")
    print(f"Diffusion Rate: {final_fitness['diffusion_rate']:.2f}%")
    print(f"Average Bit Flips: {final_fitness['avg_bit_flips']:.2f} of 32 bits")
    print(f"Total Gates: {len(best_individual)}")
    
    # Save the best solution
    with open('best_mix_column_final.pkl', 'wb') as f:
        pickle.dump({
            'gates': best_individual,
            'fitness': final_fitness
        }, f)
    
    # Generate and return the functions
    mix_columns_code = generate_mix_columns_code(best_individual)
    inverse_mix_columns_code = generate_inverse_mix_columns_code(best_individual)
    
    return best_individual, final_fitness, mix_columns_code, inverse_mix_columns_code

def generate_mix_columns_code(gate_sequence):
    code = []
    code.append("def mix_columns(circuit, qubits):")
    code.append("    for col in range(4):")
    code.append("        col_start = col * 2")
    code.append("        ")
    code.append("        c0 = col_start")
    code.append("        c8 = 8 + col_start")
    code.append("        c16 = 16 + col_start")
    code.append("        c24 = 24 + col_start")
    code.append("        ")
    
    gate_strings = []
    for gate in gate_sequence:
        gate_type = gate[0]
        if gate_type == 'cx':
            control, target = gate[1], gate[2]
            gate_strings.append(f"        circuit.cx(qubits[{control} % 8 + (col_start if {control} < 8 else 8 + col_start if {control} < 16 else 16 + col_start if {control} < 24 else 24 + col_start)], qubits[{target} % 8 + (col_start if {target} < 8 else 8 + col_start if {target} < 16 else 16 + col_start if {target} < 24 else 24 + col_start)])")
        elif gate_type == 'swap':
            q1, q2 = gate[1], gate[2]
            gate_strings.append(f"        circuit.swap(qubits[{q1} % 8 + (col_start if {q1} < 8 else 8 + col_start if {q1} < 16 else 16 + col_start if {q1} < 24 else 24 + col_start)], qubits[{q2} % 8 + (col_start if {q2} < 8 else 8 + col_start if {q2} < 16 else 16 + col_start if {q2} < 24 else 24 + col_start)])")
        elif gate_type == 'x':
            q = gate[1]
            gate_strings.append(f"        circuit.x(qubits[{q} % 8 + (col_start if {q} < 8 else 8 + col_start if {q} < 16 else 16 + col_start if {q} < 24 else 24 + col_start)])")
    
    code.extend(gate_strings)
    code.append("    ")
    code.append("    return circuit")
    
    return "\n".join(code)

def generate_inverse_mix_columns_code(gate_sequence):
    inverse_sequence = generate_inverse(gate_sequence)
    
    code = []
    code.append("def inverse_mix_columns(circuit, qubits):")
    code.append("    for col in range(3, -1, -1):")
    code.append("        col_start = col * 2")
    code.append("        ")
    code.append("        c0 = col_start")
    code.append("        c8 = 8 + col_start")
    code.append("        c16 = 16 + col_start")
    code.append("        c24 = 24 + col_start")
    code.append("        ")
    
    gate_strings = []
    for gate in inverse_sequence:
        gate_type = gate[0]
        if gate_type == 'cx':
            control, target = gate[1], gate[2]
            gate_strings.append(f"        circuit.cx(qubits[{control} % 8 + (col_start if {control} < 8 else 8 + col_start if {control} < 16 else 16 + col_start if {control} < 24 else 24 + col_start)], qubits[{target} % 8 + (col_start if {target} < 8 else 8 + col_start if {target} < 16 else 16 + col_start if {target} < 24 else 24 + col_start)])")
        elif gate_type == 'swap':
            q1, q2 = gate[1], gate[2]
            gate_strings.append(f"        circuit.swap(qubits[{q1} % 8 + (col_start if {q1} < 8 else 8 + col_start if {q1} < 16 else 16 + col_start if {q1} < 24 else 24 + col_start)], qubits[{q2} % 8 + (col_start if {q2} < 8 else 8 + col_start if {q2} < 16 else 16 + col_start if {q2} < 24 else 24 + col_start)])")
        elif gate_type == 'x':
            q = gate[1]
            gate_strings.append(f"        circuit.x(qubits[{q} % 8 + (col_start if {q} < 8 else 8 + col_start if {q} < 16 else 16 + col_start if {q} < 24 else 24 + col_start)])")
    
    code.extend(gate_strings)
    code.append("    ")
    code.append("    return circuit")
    
    return "\n".join(code)

def verify_circuit(gate_sequence, min_gates=10):
    # First check if gate sequence is valid
    if len(gate_sequence) < min_gates:
        return {
            'success_rate': 0.0,
            'diffusion_rate': 0.0,
            'avg_bit_flips': 0.0,
            'total_tests': 0
        }
    
    test_vectors = []
    
    # Add standard test cases
    test_vectors.append('0' * 32)
    test_vectors.append('1' * 32)
    
    # Add single bit set vectors
    for i in range(32):
        vector = ['0'] * 32
        vector[i] = '1'
        test_vectors.append(''.join(vector))
    
    # Add alternating patterns
    test_vectors.append(''.join(['1' if i % 2 == 0 else '0' for i in range(32)]))
    test_vectors.append(''.join(['0' if i % 2 == 0 else '1' for i in range(32)]))
    test_vectors.append(''.join(['1' if i % 4 < 2 else '0' for i in range(32)]))
    
    # Add random vectors
    for _ in range(50):
        test_vectors.append(''.join(random.choice(['0', '1']) for _ in range(32)))
    
    inverse_sequence = generate_inverse(gate_sequence)
    simulator = AerSimulator()
    
    success_count = 0
    total_bit_flips = 0
    
    for test_vector in test_vectors:
        # Forward transformation
        data_register = QuantumRegister(32, 'data')
        measurement_register = ClassicalRegister(32, 'measure')
        forward_circuit = QuantumCircuit(data_register, measurement_register)
        
        # Initialize input
        for i in range(32):
            if test_vector[i] == '1':
                forward_circuit.x(data_register[i])
        
        # Apply mix_columns
        forward_circuit = apply_gates(forward_circuit, data_register, gate_sequence)
        forward_circuit.measure(data_register, measurement_register)
        
        # Run
        forward_result = simulator.run(forward_circuit).result()
        forward_counts = forward_result.get_counts()
        output_vector = max(forward_counts, key=forward_counts.get)
        
        # Count bit changes
        bit_flips = sum(1 for a, b in zip(test_vector, output_vector) if a != b)
        total_bit_flips += bit_flips
        
        # Skip identity transformations
        if bit_flips == 0:
            continue
        
        # Inverse transformation
        inverse_circuit = QuantumCircuit(data_register, measurement_register)
        
        # Initialize with output
        for i in range(32):
            if output_vector[i] == '1':
                inverse_circuit.x(data_register[i])
        
        # Apply inverse_mix_columns
        inverse_circuit = apply_gates(inverse_circuit, data_register, inverse_sequence)
        inverse_circuit.measure(data_register, measurement_register)
        
        # Run
        inverse_result = simulator.run(inverse_circuit).result()
        inverse_counts = inverse_result.get_counts()
        recovered_vector = max(inverse_counts, key=inverse_counts.get)
        
        # Check if recovery was successful
        if recovered_vector == test_vector:
            success_count += 1
    
    success_rate = success_count / len(test_vectors) * 100
    avg_bit_flips = total_bit_flips / len(test_vectors)
    diffusion_rate = avg_bit_flips / 32 * 100
    
    return {
        'success_rate': success_rate,
        'diffusion_rate': diffusion_rate,
        'avg_bit_flips': avg_bit_flips,
        'total_tests': len(test_vectors)
    }

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Set parameters
    min_gates = 15  # Minimum number of gates to prevent trivial solutions
    
    print("Starting Mix Columns Circuit Optimization")
    print("1. Converting high diffusion circuit to gate sequence")
    base_sequence = high_diffusion_to_gate_sequence()
    
    print(f"Base circuit has {len(base_sequence)} gates")
    print("2. Evaluating base circuit")
    base_fitness = verify_circuit(base_sequence, min_gates)
    print(f"Base circuit results:")
    print(f"  - Success Rate: {base_fitness['success_rate']:.2f}%")
    print(f"  - Diffusion Rate: {base_fitness['diffusion_rate']:.2f}%")
    print(f"  - Average Bit Flips: {base_fitness['avg_bit_flips']:.2f} of 32 bits")
    
    print("\n3. Running genetic algorithm to optimize circuit")
    best_gates, best_fitness, mix_code, inverse_code = genetic_algorithm(
        base_sequence=base_sequence,
        pop_size=40, 
        generations=30,
        num_processors=4,
        min_gates=min_gates
    )
    
    print("\n4. Final optimized circuit:")
    print(f"  - Total Gates: {len(best_gates)}")
    print(f"  - Success Rate: {best_fitness['success_rate']:.2f}%")
    print(f"  - Diffusion Rate: {best_fitness['diffusion_rate']:.2f}%")
    print(f"  - Average Bit Flips: {best_fitness['avg_bit_flips']:.2f} of 32 bits")
    
    print("\n5. Saving optimized mix_columns function:")
    with open('optimized_mix_columns.py', 'w') as f:
        f.write(mix_code + "\n\n" + inverse_code)
    
    print("Saved to optimized_mix_columns.py")
    
    print("\nDone! The generated functions can be integrated into your quantum circuit.")

if __name__ == "__main__":
    main()