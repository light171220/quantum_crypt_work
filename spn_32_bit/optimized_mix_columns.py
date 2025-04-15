def mix_columns(circuit, qubits):
    for col in range(4):
        col_start = col * 2
        
        c0 = col_start
        c8 = 8 + col_start
        c16 = 16 + col_start
        c24 = 24 + col_start
        
        circuit.cx(qubits[9 % 8 + (col_start if 9 < 8 else 8 + col_start if 9 < 16 else 16 + col_start if 9 < 24 else 24 + col_start)], qubits[8 % 8 + (col_start if 8 < 8 else 8 + col_start if 8 < 16 else 16 + col_start if 8 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[12 % 8 + (col_start if 12 < 8 else 8 + col_start if 12 < 16 else 16 + col_start if 12 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[7 % 8 + (col_start if 7 < 8 else 8 + col_start if 7 < 16 else 16 + col_start if 7 < 24 else 24 + col_start)])
        circuit.cx(qubits[18 % 8 + (col_start if 18 < 8 else 8 + col_start if 18 < 16 else 16 + col_start if 18 < 24 else 24 + col_start)], qubits[31 % 8 + (col_start if 31 < 8 else 8 + col_start if 31 < 16 else 16 + col_start if 31 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[18 % 8 + (col_start if 18 < 8 else 8 + col_start if 18 < 16 else 16 + col_start if 18 < 24 else 24 + col_start)], qubits[31 % 8 + (col_start if 31 < 8 else 8 + col_start if 31 < 16 else 16 + col_start if 31 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[30 % 8 + (col_start if 30 < 8 else 8 + col_start if 30 < 16 else 16 + col_start if 30 < 24 else 24 + col_start)], qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)])
        circuit.cx(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)])
        circuit.cx(qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.cx(qubits[30 % 8 + (col_start if 30 < 8 else 8 + col_start if 30 < 16 else 16 + col_start if 30 < 24 else 24 + col_start)], qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)])
        circuit.cx(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)])
        circuit.cx(qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.swap(qubits[27 % 8 + (col_start if 27 < 8 else 8 + col_start if 27 < 16 else 16 + col_start if 27 < 24 else 24 + col_start)], qubits[10 % 8 + (col_start if 10 < 8 else 8 + col_start if 10 < 16 else 16 + col_start if 10 < 24 else 24 + col_start)])
        circuit.swap(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.swap(qubits[27 % 8 + (col_start if 27 < 8 else 8 + col_start if 27 < 16 else 16 + col_start if 27 < 24 else 24 + col_start)], qubits[10 % 8 + (col_start if 10 < 8 else 8 + col_start if 10 < 16 else 16 + col_start if 10 < 24 else 24 + col_start)])
        circuit.swap(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
    
    return circuit

def inverse_mix_columns(circuit, qubits):
    for col in range(3, -1, -1):
        col_start = col * 2
        
        c0 = col_start
        c8 = 8 + col_start
        c16 = 16 + col_start
        c24 = 24 + col_start
        
        circuit.swap(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.swap(qubits[27 % 8 + (col_start if 27 < 8 else 8 + col_start if 27 < 16 else 16 + col_start if 27 < 24 else 24 + col_start)], qubits[10 % 8 + (col_start if 10 < 8 else 8 + col_start if 10 < 16 else 16 + col_start if 10 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.swap(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.swap(qubits[27 % 8 + (col_start if 27 < 8 else 8 + col_start if 27 < 16 else 16 + col_start if 27 < 24 else 24 + col_start)], qubits[10 % 8 + (col_start if 10 < 8 else 8 + col_start if 10 < 16 else 16 + col_start if 10 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)])
        circuit.cx(qubits[30 % 8 + (col_start if 30 < 8 else 8 + col_start if 30 < 16 else 16 + col_start if 30 < 24 else 24 + col_start)], qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[23 % 8 + (col_start if 23 < 8 else 8 + col_start if 23 < 16 else 16 + col_start if 23 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[22 % 8 + (col_start if 22 < 8 else 8 + col_start if 22 < 16 else 16 + col_start if 22 < 24 else 24 + col_start)], qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)])
        circuit.cx(qubits[30 % 8 + (col_start if 30 < 8 else 8 + col_start if 30 < 16 else 16 + col_start if 30 < 24 else 24 + col_start)], qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[18 % 8 + (col_start if 18 < 8 else 8 + col_start if 18 < 16 else 16 + col_start if 18 < 24 else 24 + col_start)], qubits[31 % 8 + (col_start if 31 < 8 else 8 + col_start if 31 < 16 else 16 + col_start if 31 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[15 % 8 + (col_start if 15 < 8 else 8 + col_start if 15 < 16 else 16 + col_start if 15 < 24 else 24 + col_start)])
        circuit.cx(qubits[18 % 8 + (col_start if 18 < 8 else 8 + col_start if 18 < 16 else 16 + col_start if 18 < 24 else 24 + col_start)], qubits[31 % 8 + (col_start if 31 < 8 else 8 + col_start if 31 < 16 else 16 + col_start if 31 < 24 else 24 + col_start)])
        circuit.cx(qubits[6 % 8 + (col_start if 6 < 8 else 8 + col_start if 6 < 16 else 16 + col_start if 6 < 24 else 24 + col_start)], qubits[7 % 8 + (col_start if 7 < 8 else 8 + col_start if 7 < 16 else 16 + col_start if 7 < 24 else 24 + col_start)])
        circuit.cx(qubits[14 % 8 + (col_start if 14 < 8 else 8 + col_start if 14 < 16 else 16 + col_start if 14 < 24 else 24 + col_start)], qubits[12 % 8 + (col_start if 12 < 8 else 8 + col_start if 12 < 16 else 16 + col_start if 12 < 24 else 24 + col_start)])
        circuit.cx(qubits[9 % 8 + (col_start if 9 < 8 else 8 + col_start if 9 < 16 else 16 + col_start if 9 < 24 else 24 + col_start)], qubits[8 % 8 + (col_start if 8 < 8 else 8 + col_start if 8 < 16 else 16 + col_start if 8 < 24 else 24 + col_start)])
    
    return circuit