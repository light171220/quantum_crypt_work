from qiskit import QuantumCircuit

def s0_box_circuit():
    qc = QuantumCircuit(4)

    qc.x(0)
    qc.x(0)
    qc.swap(3, 1)
    qc.cx(3, 2)
    qc.ccx(1, 2, 3)
    qc.swap(0, 1)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 1, 3)
    qc.cx(1, 3)
    qc.ccx(0, 2, 3)
    qc.swap(1, 3)
    qc.swap(1, 0)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.cx(2, 1)

    return qc

qc = s0_box_circuit()
print(qc.draw())
