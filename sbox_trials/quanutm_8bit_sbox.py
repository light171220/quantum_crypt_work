from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CXGate, SwapGate

def aes_sbox_with_swaps():
    # Registers (20 qubits: 8 input, 8 output, 4 auxiliary)
    qr_a = QuantumRegister(8, 'a')      # Input byte
    qr_s = QuantumRegister(8, 'S(a)')   # Output S-box
    qr_aux = QuantumRegister(4, 'aux')  # Auxiliary qubits
    qc = QuantumCircuit(qr_a, qr_s, qr_aux, name="AES_SBOX_SWAPS")

    # --- STEP 1: Matrix Multiplication (U_M) with Swaps (Fig. 2) ---
    # Apply CNOTs and SWAPs to compute |Ma⟩
    qc.cx(qr_a[7], qr_s[0])
    qc.cx(qr_a[6], qr_s[1])
    qc.cx(qr_a[5], qr_s[2])
    qc.cx(qr_a[4], qr_s[3])
    qc.swap(qr_a[3], qr_s[4])  # Swap instead of CNOT
    qc.swap(qr_a[2], qr_s[5])
    qc.swap(qr_a[1], qr_s[6])
    qc.swap(qr_a[0], qr_s[7])

    # Continue with remaining CNOTs (adjust indices due to swaps)
    qc.cx(qr_s[7], qr_s[1])
    qc.cx(qr_s[6], qr_s[2])
    qc.cx(qr_s[5], qr_s[3])
    qc.cx(qr_s[4], qr_s[4])  # No-op (self-loop)
    qc.cx(qr_s[3], qr_s[5])
    qc.cx(qr_s[2], qr_s[6])
    qc.cx(qr_s[1], qr_s[7])

    # --- STEP 2: Multiplicative Inversion (U_inv0) ---
    # (Same as before, but account for swapped qubits)
    p0 = [qr_s[i] for i in range(4)]  # p0 = s[0..3]
    p1 = [qr_s[i] for i in range(4,8)] # p1 = s[4..7]

    # Compute p1^2 * λ (Eq. 6)
    qc.cx(p1[1], qr_aux[0])
    qc.cx(p1[0], qr_aux[0])  # aux[0] = p1[1] + p1[0]
    qc.cx(p1[3], qr_aux[1])
    qc.cx(p1[1], qr_aux[1])
    qc.cx(p1[0], qr_aux[1])  # aux[1] = p1[3] + p1[1] + p1[0]
    qc.cx(p1[0], qr_aux[2])  # aux[2] = p1[0]
    qc.cx(p1[2], qr_aux[3])
    qc.cx(p1[1], qr_aux[3])  # aux[3] = p1[2] + p1[1]

    # --- STEP 3: Affine Transform (U_AM^{-1}) with Swaps (Fig. 3) ---
    qc.swap(qr_s[7], qr_s[0])  # Swap instead of CNOT
    qc.swap(qr_s[6], qr_s[1])
    qc.swap(qr_s[5], qr_s[2])
    qc.swap(qr_s[4], qr_s[3])

    # Remaining CNOTs
    qc.cx(qr_s[3], qr_s[4])
    qc.cx(qr_s[2], qr_s[5])
    qc.cx(qr_s[1], qr_s[6])
    qc.cx(qr_s[0], qr_s[7])

    # --- STEP 4: Add Constant c ---
    qc.x(qr_s[0])
    qc.x(qr_s[1])
    qc.x(qr_s[5])
    qc.x(qr_s[6])

    return qc

# Generate and draw the circuit
qc_swapped = aes_sbox_with_swaps()
print(qc_swapped.draw(fold=-1))