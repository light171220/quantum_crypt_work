import numpy as np
from collections import defaultdict

class QuantumRegister:
    def __init__(self, size, name):
        self.size = size
        self.name = name
        self.qubits = list(range(size))
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.size)
            return [Qubit(self, i) for i in range(start, stop, step)]
        return Qubit(self, key)
    
    def __len__(self):
        return self.size

class Qubit:
    def __init__(self, register, index):
        self.register = register
        self.index = index
        
    def __repr__(self):
        return f"{self.register.name}[{self.index}]"

class ClassicalRegister:
    def __init__(self, size, name):
        self.size = size
        self.name = name
        self.bits = [0] * size
    
    def __getitem__(self, key):
        return self.bits[key]

class QuantumCircuit:
    def __init__(self, *registers, name=None):
        self.quantum_registers = []
        self.classical_registers = []
        
        for reg in registers:
            if isinstance(reg, QuantumRegister):
                self.quantum_registers.append(reg)
            elif isinstance(reg, ClassicalRegister):
                self.classical_registers.append(reg)
        
        self.name = name
        self.instructions = []
        
        # Calculate total qubits
        self.num_qubits = sum(reg.size for reg in self.quantum_registers)
        
        # Map to reference qubits by index
        self.qubit_map = {}
        index = 0
        for reg in self.quantum_registers:
            for i in range(reg.size):
                self.qubit_map[(reg.name, i)] = index
                index += 1
    
    def _get_qubit_index(self, qubit):
        if isinstance(qubit, Qubit):
            return self.qubit_map[(qubit.register.name, qubit.index)]
        return qubit  # Assume it's already an index
    
    def _add_instruction(self, name, qubits, params=None):
        qubit_indices = [self._get_qubit_index(q) for q in qubits]
        self.instructions.append((name, qubit_indices, params))
    
    def h(self, qubit):
        self._add_instruction('h', [qubit])
        return self
    
    def x(self, qubit):
        self._add_instruction('x', [qubit])
        return self
    
    def cx(self, control, target):
        self._add_instruction('cx', [control, target])
        return self
    
    def ccx(self, control1, control2, target):
        self._add_instruction('ccx', [control1, control2, target])
        return self
    
    def mcx(self, controls, target):
        if isinstance(controls, list):
            self._add_instruction('mcx', controls + [target])
        else:
            self._add_instruction('mcx', [controls, target])
        return self
    
    def measure(self, qubits, classical_bits=None):
        if isinstance(qubits, QuantumRegister) and isinstance(classical_bits, ClassicalRegister):
            for i in range(min(len(qubits), len(classical_bits.bits))):
                self._add_instruction('measure', [qubits[i]], {'classical_bit': (classical_bits.name, i)})
        elif isinstance(qubits, list) and isinstance(classical_bits, ClassicalRegister):
            for i, qubit in enumerate(qubits):
                if i < len(classical_bits.bits):
                    self._add_instruction('measure', [qubit], {'classical_bit': (classical_bits.name, i)})
        else:
            self._add_instruction('measure', [qubits], {'classical_bit': classical_bits})
        return self
    
    def barrier(self):
        self._add_instruction('barrier', [])
        return self
    
    def swap(self, qubit1, qubit2):
        self._add_instruction('swap', [qubit1, qubit2])
        return self

class SimpleQuantumSimulator:
    
    def __init__(self, method='statevector'):
        self.method = method
        self.state = None
        self.shots = 1024  # Default
        
    def _initialize_state(self, num_qubits):
        dim = 2**num_qubits
        self.state = np.zeros(dim, dtype=complex)
        self.state[0] = 1.0  # |00...0⟩ state
        
    def _apply_gate(self, gate_type, qubits, num_qubits):
        if gate_type == 'x':
            target = qubits[0]
            # Apply X gate (bit flip)
            for i in range(2**num_qubits):
                if (i >> target) & 1:  # if target bit is 1
                    idx_to_swap = i & ~(1 << target)  # flip target bit to 0
                else:
                    idx_to_swap = i | (1 << target)  # flip target bit to 1
                
                # Swap amplitudes
                self.state[i], self.state[idx_to_swap] = self.state[idx_to_swap], self.state[i]
        
        elif gate_type == 'h':
            target = qubits[0]
            # Apply Hadamard gate
            new_state = np.zeros_like(self.state)
            
            for i in range(2**num_qubits):
                bit = (i >> target) & 1
                if bit == 0:
                    # If target bit is 0, add amplitude to both 0 and 1 states
                    new_state[i] += self.state[i] / np.sqrt(2)
                    new_state[i | (1 << target)] += self.state[i] / np.sqrt(2)
                else:
                    # If target bit is 1, add amplitude to 0 state and subtract from 1 state
                    new_state[i & ~(1 << target)] += self.state[i] / np.sqrt(2)
                    new_state[i] -= self.state[i] / np.sqrt(2)
            
            self.state = new_state
        
        elif gate_type == 'cx':
            control, target = qubits
            # Apply controlled-X gate
            for i in range(2**num_qubits):
                if (i >> control) & 1:  # if control bit is 1
                    if (i >> target) & 1:  # if target bit is 1
                        idx_with_target_0 = i & ~(1 << target)  # same state but with target=0
                        self.state[i], self.state[idx_with_target_0] = self.state[idx_with_target_0], self.state[i]
                    else:  # if target bit is 0
                        idx_with_target_1 = i | (1 << target)  # same state but with target=1
                        self.state[i], self.state[idx_with_target_1] = self.state[idx_with_target_1], self.state[i]
        
        elif gate_type == 'ccx' or gate_type == 'mcx':
            # For ccx (Toffoli) the controls are the first two qubits, target is the last
            # For mcx, controls are all but the last qubit
            controls = qubits[:-1]
            target = qubits[-1]
            
            # Check if all control bits are 1, then flip the target
            for i in range(2**num_qubits):
                # Check if all control bits are 1
                all_controls_one = True
                for control in controls:
                    if not ((i >> control) & 1):
                        all_controls_one = False
                        break
                
                if all_controls_one:
                    # Flip target bit
                    if (i >> target) & 1:  # if target bit is 1
                        idx_with_target_0 = i & ~(1 << target)  # flip to 0
                        self.state[i], self.state[idx_with_target_0] = self.state[idx_with_target_0], self.state[i]
                    else:  # if target bit is 0
                        idx_with_target_1 = i | (1 << target)  # flip to 1
                        self.state[i], self.state[idx_with_target_1] = self.state[idx_with_target_1], self.state[i]
        
        elif gate_type == 'swap':
            qubit1, qubit2 = qubits
            # Apply SWAP gate
            for i in range(2**num_qubits):
                bit1 = (i >> qubit1) & 1
                bit2 = (i >> qubit2) & 1
                
                if bit1 != bit2:  # Only swap if bits are different
                    # Calculate the state index with these bits swapped
                    swapped_i = i ^ (1 << qubit1) ^ (1 << qubit2)
                    self.state[i], self.state[swapped_i] = self.state[swapped_i], self.state[i]
    
    def _measure_state(self, num_qubits, shots):
        # Calculate probabilities from amplitudes
        probabilities = np.abs(self.state)**2
        
        # Sample from the probability distribution
        results = np.random.choice(2**num_qubits, size=shots, p=probabilities)
        
        # Count occurrences and format as binary strings
        counts = defaultdict(int)
        for result in results:
            # Format as binary string, padding with leading zeros
            binary = format(result, f'0{num_qubits}b')
            counts[binary] += 1
        
        return dict(counts)
    
    def run(self, circuit, shots=1024):
        self.shots = shots
        num_qubits = circuit.num_qubits
        
        # Initialize quantum state
        self._initialize_state(num_qubits)
        
        # Apply circuit instructions
        for gate_type, qubits, params in circuit.instructions:
            if gate_type == 'measure':
                continue  # Skip for now, we'll handle measurements at the end
            elif gate_type == 'barrier':
                continue  # Barriers have no effect on simulation
            else:
                self._apply_gate(gate_type, qubits, num_qubits)
        
        # Perform measurement
        counts = self._measure_state(num_qubits, shots)
        
        return SimpleQuantumResult(counts)

class SimpleQuantumResult:
    
    def __init__(self, counts):
        self.counts = counts
    
    def get_counts(self):
        return self.counts

# Example usage functions that work with our new simulator
def test_simple_circuit():
    qreg = QuantumRegister(2, 'q')
    creg = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qreg, creg)
    
    # Create a Bell state
    circuit.h(qreg[0])
    circuit.cx(qreg[0], qreg[1])
    circuit.measure(qreg, creg)
    
    # Run the circuit
    simulator = SimpleQuantumSimulator()
    result = simulator.run(circuit, shots=1024)
    counts = result.get_counts()
    
    # Print results
    print("Measurement results for Bell state:")
    for state, count in counts.items():
        print(f"|{state}>: {count} ({count/1024:.4f})")
    
    return counts