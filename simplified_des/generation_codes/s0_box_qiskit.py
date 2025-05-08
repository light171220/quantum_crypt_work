from qiskit import QuantumCircuit

def s0_box_circuit():
    qc = QuantumCircuit(4)

    qc.ccx(0, 1, 3)
    qc.ccx(0, 2, 3)
    qc.ccx(0, 1, 2)
    qc.swap(2, 0)
    qc.swap(0, 2)
    qc.cx(3, 1)
    qc.x(2)
    qc.ccx(0, 1, 2)
    qc.cx(2, 0)

    return qc

qc = s0_box_circuit()
print(qc.draw())
