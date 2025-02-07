import qiskit as qk
import numpy as np
import scipy.linalg as lin
import math
from qiskit.quantum_info import Statevector , Operator , partial_trace , DensityMatrix
from qiskit.circuit.library.standard_gates import HGate , XGate
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.extensions import Initialize , UnitaryGate
from qiskit.providers.aer import AerSimulator

def diagonal_unitary(DiagonalParams , EvolutionTime):
    """
    
    Input #1 (hs): an array f doubles hopping strengths 
    Input #2 (Vs): a coupling strength V for all the XX interactions
    Input #3 (ev_time): the time of evolution

    output: a unitary gate simulating the diagonal evolution U_0(t = EvolutionTime)

    """

    hs = DiagonalParams[0]
    Vx = DiagonalParams[1]

    # The number of spins is specified by the number of diagonal h_i Z_i terms
    NumberOfSpins = len(hs)
    U0 = qk.QuantumCircuit(NumberOfSpins)

    for x in range(NumberOfSpins):
        U0.rz(2*hs[x]*EvolutionTime , x)

    # For two-body ZZ interactions (with V_x):
    NumberOfPermutations = len(Vx)
    for x in range(NumberOfPermutations):
        U0.cx(x , x+1)
        U0.rz(2*Vx[x]*EvolutionTime , x+1)
        U0.cx(x , x+1)
    return U0.to_gate()

# =================================================================================== #
# ------------------------------ LCU State preparation ------------------------------ #

def Q_angle(Q_max , GammaDt , i):
    angle = 0.0
    for q in np.arange(1 , Q_max - i+1):
        angle += GammaDt**(q) / math.factorial(q + i)
    angle *= math.factorial(i)
    angle = np.sqrt(angle)

    return np.arctan(angle)


def UQ_circuit(Q_max , number_of_spins , GammaDt , dagger=False):
    L = int(np.log2(number_of_spins))
    uq = qk.QuantumCircuit(Q_max*L)
    if dagger:
        for q in range(Q_max-1 , 0 , -1):
            uq.cry(-2.0*Q_angle(Q_max , GammaDt , q) , (q-1)*L , q*L)
        uq.ry(-2.0*Q_angle(Q_max , GammaDt , 0) , 0)  
    else:
        uq.ry(2.0*Q_angle(Q_max , GammaDt , 0) , 0)
        for q in np.arange(1,Q_max):
            uq.cry(2.0*Q_angle(Q_max , GammaDt , q) , (q-1)*L , q*L) 
    return uq

# Creating a unitary matrix out of a column vector using the Householder matrix
def create_unitary(v):
    dim = v.size
    # Return identity if v is a multiple of e1
    if v[0][0] and not np.any(v[0][1:]):
        return np.identity( dim )
    e1 = np.zeros( dim )
    e1[0] = 1
    w = v/np.linalg.norm(v) - e1
    return np.identity(dim) - 2*((np.dot(w.T, w))/(np.dot(w, w.T)))

def create_unitary_from_vector(vec, dagger=False):
    U = create_unitary(vec)
    if dagger:
        U = U.conj().T
    return UnitaryGate(U)

def create_unitary_map_for_permutation_states(vec , dagger=False):
    N = vec[0].size

    U = np.zeros((N+1,N+1))
    U[0][0] = 1.0
    
    unit = create_unitary(vec)
    U[1: , 1:] = unit
    if dagger:
        U = U.conj().T
    return UnitaryGate(U)

def B_state_prepare(Circuit , Q_max , Delta_t , Gamma_list , kq_qbits_idx , iq_qbits_idx , q_qbits_idx ,  dagger=False):
        
    """
    This is a void function preparing the states on |i_q> , |k_q> , and |q> registers.

    Input (QuantumCircuit) #1: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (int) #2 (Q_max): The maximum expansion order Q
    Input (float) #3 (GammaDt): The parameter Gamma*Delta_t that specifies the simulation parameter.
    Input (fnc) #4 (Gamma_function): The function specifying the off-diagonal coefficients of the Hamiltonian
    Input (int) #5 (kq_qbits_idx): The index of the register of the |k_q> qubits
    Input (int) #6 (iq_qbits_idx): The index of the register of the |i_q> qubits
    Input (int) #7 (q_qbits_idx): The index of the register of the |q> qubits
    Input (bool) #8 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 
    
    """
    Gamma_i = Gamma_list[0]
    Gamma_k = Gamma_list[1]
    M = len(Gamma_i)
    K = len(Gamma_k)

    if M < 2:
        LM = 1
    else:
        Log2M = np.log2(M)
        LM = int(np.ceil(Log2M))
    MDisc = int(2**LM - M)-1 

    if K < 2:
        LK = 1
    else:
        Log2K = np.log2(K)
        LK = int(np.ceil(Log2K))
    KDisc = int(2**LK - K) 

    iq_qbits = Circuit.qregs[iq_qbits_idx]
    kq_qbits = Circuit.qregs[kq_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]

    Gamma_1 = np.sum(Gamma_i)
    Gamma_2 = np.sum(Gamma_k)
    Gamma = Gamma_1 * Gamma_2 
    GDt = Gamma*Delta_t
    Q = Q_max

    # Generating the B_k rotation:
    MappingVecBk = np.zeros((1 , int(2**LK)))
    if KDisc == 0:
        for i in range(K):
            MappingVecBk[0][i] = np.sqrt( Gamma_k[i]/Gamma_2 )
    else:
        MappingVecBk[0][0] = 0
        for i in np.arange(1 , K+1):
            MappingVecBk[0][i] = np.sqrt( Gamma_k[i-1]/Gamma_2 )

    # The B_k rotations are controlled rotations!
    # These rotations dont need to be controlled! CNOT_j U CNOT_j can be applied CU_j! 
    UmapBkdag = create_unitary_from_vector(MappingVecBk , True).control(1)
    UmapBk = create_unitary_from_vector(MappingVecBk , False).control(1)


    # Generating the B_i(2) rotation:
    # It is assumed that M < 2**n , where n is an integer!
    MappingVecBi2 = np.zeros((1 , int(2**LM) - 1))
    for i in range(M):
        MappingVecBi2[0][i+MDisc] = np.sqrt( Gamma_i[i] / Gamma_1 )

    UmapBi2dag = create_unitary_map_for_permutation_states(MappingVecBi2 , True)
    UmapBi2 = create_unitary_map_for_permutation_states(MappingVecBi2 , False)
    
    if dagger:
        for i in range(Q-1 , -1 , -1):
            # If M = 1, then there is only one type of permutation, so the additional rotation is not required!
            if M > 1:
                Circuit.append(UmapBi2dag , iq_qbits[list(np.arange(i*LM , (i+1)*LM))])
        # Applying controlled B_k rotations
        for i in range(Q-1 , -1 , -1):
            Circuit.append(UmapBkdag , [iq_qbits[i*LM] , kq_qbits[list(np.arange(i*LK , (i+1)*LK))]])
            Circuit.cx(iq_qbits[i*LM] , q_qbits[i])
        # Applying the initial rotations on |i_q> register (uncompute):
        for q in range(Q-1 , 0 , -1):
            Circuit.cry(-2.0*Q_angle(Q , GDt , q) , iq_qbits[(q-1)*LM] , iq_qbits[q*LM])
        Circuit.ry(-2.0*Q_angle(Q , GDt , 0) , iq_qbits[0])
    else:
        # Applying the initial rotations on the |i_q> register:
        Circuit.ry(2.0*Q_angle(Q , GDt , 0) , iq_qbits[0]) 
        for q in np.arange(1, Q):
            Circuit.cry(2.0*Q_angle(Q , GDt , q) , iq_qbits[(q-1)*LM] , iq_qbits[q*LM])
        # Applying the controlled B_k rotations:
        for i in range(Q):
            Circuit.cx(iq_qbits[i*LM] , q_qbits[i])
            #Circuit.append(UmapBk ,  kq_qbits[list(np.arange(i*LK , (i+1)*LK))])
            Circuit.append(UmapBk , [iq_qbits[i*LM] , kq_qbits[list(np.arange(i*LK , (i+1)*LK))]])
        for i in range(Q):
            # If M = 1, then there is only one type of permutation, so the additional rotation is not required!
            if M > 1:
                Circuit.append(UmapBi2 , iq_qbits[list(np.arange(i*LM , (i+1)*LM))])


# =================================================================================== #
# ------------------------ Controlled Unitary functions ------------------------ #

# ================================= The controlled unitary Uc_P =============================== #
# Adopting the convention of labels: if P_i represents X_i X_i+1, e.g. |i=5> corresponds to X_5 X_6 permutation

# To build a generalized controlled unitary operator, we need to specify a set of controlled qubits (q_c),
#    and act upon these controlled unitary qubits with a sequence of controlled permutations X_i X_i+1 s

# Defining the function Uc_P to produce controlled unitary gates on a set of control qubits and target qubits!

# ctr_qbits_idx will be the index for the set of quantum registers of the Circuit
#    that act as a control qubit.. Same for targ_qbits_idx

# Note that iq_qbits register has Q*log2(n) number of qubits:
#   The SubIndex is a variable to specify the permutation for the specific sub index of the iq_qbits

def Uc_P( Circuit , NumOfPermutations , CtrQbitsIndex , TargQbitsIndex , SubIndex , dagger):
    """"
    This is a void function creating a controlled permutation on the circuit

    Input (int) #1 (NumOfPermutations): The number of spins for the system
    Input (int) #2 (CtrQbitsIndex): The index of the register for the controlled qubits
    Input (int) #3 (TargQbitsIndex): The index of the register for the target qubits
    Input (int) #4 (SubIndex): The sub-index (q) for the i_q register to generate the controlled qubit
    Input (bool) #5 (dagger): Speficies whether the gate is hermitian conjugate of U_p or not
    """
    # Assuming SubIndex refers to a specific block of register of size log2(n) qubits
    if SubIndex > 0:
        CtrQbits = Circuit.qregs[CtrQbitsIndex]
        TargQbits = Circuit.qregs[TargQbitsIndex]
        L = int( np.ceil(np.log2(NumOfPermutations)) )
        NQbits = Circuit.num_qubits # Total qubits for the circuit
        
        CtrQbitsSubset = CtrQbits[(SubIndex-1)*L:SubIndex*L]
        # Making the controlled XX gate for the specific number of controlled qubits
        cxxcirc = qk.QuantumCircuit(2 , name='cXX')
        cxxcirc.x( range(2) )
        cxxGate = cxxcirc.to_gate()
        cxxGate = cxxGate.control(L)
        
        if dagger:
            for i in np.arange(NumOfPermutations-1 , 0 , -1):
                ibin = bin(i)[2:]
                ibin = (L-len(ibin))*'0' + ibin
                ibin = ibin[::-1]
                
                ZeroCtrls = [x for x in range(L) if(int(ibin[x]) == 0)]
                for j in range(len(ZeroCtrls)):
                    Circuit.x( CtrQbitsSubset[ZeroCtrls[j]] )
                Circuit.append( cxxGate , CtrQbitsSubset[:] + TargQbits[i-1:i+1] )
                for j in np.arange(len(ZeroCtrls)-1 , -1 , -1):
                    Circuit.x( CtrQbitsSubset[ZeroCtrls[j]] )
        else:
            for i in np.arange(1, NumOfPermutations+1):
                ibin = bin(i)[2:]
                ibin = (L-len(ibin))*'0' +ibin 
                ibin = ibin[::-1]

                ZeroCtrls = [x for x in range(L) if(int(ibin[x]) == 0)]
                for j in range(len(ZeroCtrls)):
                    Circuit.x( CtrQbitsSubset[ZeroCtrls[j]] )
                # This part needs to be fixed.. We need to 
                Circuit.append( cxxGate , CtrQbitsSubset[:] + TargQbits[i-1:i+1] )
                for j in range(len(ZeroCtrls)):
                    Circuit.x( CtrQbitsSubset[ZeroCtrls[j]] )


def U0_q_ctrl( Circuit , DiagonalParams , delta_t , q_qbits , z_qbits , dagger=False):
    
    """"

    This is a void function creating the q-dependent controlled diagonal rotations 

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (double) #1 (delta_t): Specifying the rotation amount, for which a U_0( delta_t/(q+1) ) will be applied 
    Input (int) #2 (q_qbits): The quantum register (not the index) of the q qubits
    Input (int) #3 (z_qbits): The quantum register of the z qubits
    Input (bool) #4 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 

    """

    #Longit_h = DiagonalParams[0]
    #Vx = DiagonalParams[1]

    if dagger:
        for k in range(len(q_qbits)-1 , 0 , -1):
            CUk = diagonal_unitary( DiagonalParams , delta_t/((k+1)*(k+2)) )
            CUk = CUk.control()
            Circuit.append( CUk , [q_qbits[k]] + z_qbits[:] )
        CU0 = diagonal_unitary( DiagonalParams , -delta_t/(2.0) )
        CU0 = CU0.control()
        Circuit.append( CU0 , [q_qbits[0]] + z_qbits[:] )
    else:
        CU0 = diagonal_unitary( DiagonalParams , delta_t/(2.0) )
        CU0 = CU0.control()
        Circuit.append( CU0 , [q_qbits[0]] + z_qbits[:] )
        for k in np.arange(1, len(q_qbits)):
            CUk = diagonal_unitary( DiagonalParams , -delta_t/((k+1)*(k+2)) )
            CUk = CUk.control()
            Circuit.append( CUk , [q_qbits[k]] + z_qbits[:] )


def Uc_Phi_Omega(Circuit , Omega , Delta_t , current_time , k_qbits_idx , q_qbits_idx , dagger= False):
    
    """"

    This is a void function creating the omega dependent rotations 

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (double) #1 (Omega): This is a specific parameter assuming that the time-dependent portion of the off-diagonal component is either cos(omega t) or sin(omega t)
    Input (double) #2 (Delta_t): This specifies the time step of each simulation operator
    Input (double) #3 (current_time): Specifying the current time
    Input (int) #4 (k_qbits_idx): The index of the register for the k qubits
    Input (int) #5 (q_qbits_idx): The sub-index (q) for the i_q register to generate the controlled qubit
    Input (bool) #6 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 

    """

    k_qbits = Circuit.qregs[k_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]
    
    delta = -2*Omega*(Delta_t) # Dividing the angle by 2 for rz
    phase = -2*Omega*current_time

    if dagger:
        for l in range(len(k_qbits)-1 , - 1 , -1):
            if l > 0:
                for lk in range(l , 0 , -1):
                    Circuit.crz( delta/((lk+2)*(lk+1)) , q_qbits[l] , k_qbits[l-lk])
            Circuit.crz( -(l+1)*delta/(l+2) , q_qbits[l]  , k_qbits[l])
            Circuit.crz(-1.0*phase , q_qbits[l] , k_qbits[l])
    else:
        for l in range(len(k_qbits)):
            Circuit.crz(phase , q_qbits[l] , k_qbits[l])
            Circuit.crz((l+1)*delta/(l+2) , q_qbits[l] , k_qbits[l])
            if l > 0:
                for lk in np.arange(1 , l+1):
                    Circuit.crz(-delta/((lk+2)*(lk+1)) , q_qbits[l] , k_qbits[l - lk])


# UC_Phi(iq_qbits , z_qbits , q) generates the E_z_iq and E_z_ij related phases on |z>
def Uc_Phi(Circuit , NumberOfPermutations , DiagonalParams , Delta_t , Q , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger=False):
    
    """

    This is a void function creating the q-dependent controlled rotations due to the off-diagonal expansion.

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (int) #1 (iq_qbits_idx): The index of the register of the |i_q> qubits
    Input (int) #2 (z_qbits_idx): The index of the register of the |z> qubits
    Input (int) #3 (q_qbits_idx): The index of the register of the |q> qubits
    Input (bool) #4 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 
    
    """

    M = NumberOfPermutations
    iq_qbits = Circuit.qregs[iq_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]
    z_qbits = Circuit.qregs[z_qbits_idx]
    liq = iq_qbits.size
    if dagger:
        #Circuit.append( Diagonal_U0( Longit_h , Vx , Delta_t ) , z_qbits )
        U0_q_ctrl( Circuit , DiagonalParams , -Delta_t*Q , q_qbits , z_qbits , True)
        for j in range(Q , 0 , -1):
            Uc_P( Circuit , M , iq_qbits_idx , z_qbits_idx , j , dagger )
            U0_q_ctrl( Circuit , DiagonalParams , Delta_t , q_qbits , z_qbits , True )

            Circuit.cp( np.pi , q_qbits[j-1] , z_qbits[0] )
            Circuit.crz( -np.pi , q_qbits[j-1] , z_qbits[0] )
    else:
        for j in np.arange(1,Q+1):
            Circuit.crz( np.pi , q_qbits[j-1] , z_qbits[0] )
            Circuit.cp( -np.pi , q_qbits[j-1] , z_qbits[0] )

            U0_q_ctrl( Circuit , DiagonalParams , Delta_t , q_qbits , z_qbits , False )
            Uc_P( Circuit , M , iq_qbits_idx , z_qbits_idx , j , dagger )
        U0_q_ctrl( Circuit , DiagonalParams , -Delta_t*Q , q_qbits , z_qbits , False )


# =================================================================================== #
# ---------------------- Generating the off-diagonal unitary ------------------------ #

def W_gate(Circuit , current_time , DiagonalParams , Delta_t , Omega , Gamma_list , RegisterIndices , dagger = False):
    [kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , anc_qbits_idx] = RegisterIndices
    zRegister = Circuit.qregs[z_qbits_idx]
    qRegister = Circuit.qregs[q_qbits_idx]
    Q_max = qRegister.size
    NumberOfSpins = zRegister.size - 1  # One of the qubits in the zRegister is used as an ancilla! We might be able to remove this as it is unnecessary.
    NumberOfPermutations = len(Gamma_list[0])

    B_state_prepare(Circuit , Q_max , Delta_t , Gamma_list , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , False )
    Uc_Phi( Circuit , NumberOfPermutations , DiagonalParams , Delta_t , Q_max , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger )
    Uc_Phi_Omega( Circuit , Omega , Delta_t , current_time , kq_qbits_idx , q_qbits_idx , dagger )
    B_state_prepare( Circuit , Q_max , Delta_t , Gamma_list , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , True )

# =================================================================================== #
# ------------------------ Amplitude Amplification functions ------------------------ #

# ---------------- The REFLECTION (R) about the zero state of Q+1 registers -----------
# Note: We may have to do this on (Q+1)*log2(M) registers instead to
#      make things for convenient to use

def R_gate(Circuit , RegisterIndices):
    [kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , anc_qbits_idx] = RegisterIndices
    kq_qbits = Circuit.qregs[kq_qbits_idx]
    iq_qbits = Circuit.qregs[iq_qbits_idx]
    anc_qbit = Circuit.qregs[anc_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]
    for i in range(len(kq_qbits)):
        Circuit.x(kq_qbits[i])
    for i in range(len(iq_qbits)):
        Circuit.x(iq_qbits[i])
    for i in range(len(q_qbits)):
        Circuit.x(q_qbits[i])
    Circuit.x(anc_qbit[0])
    
    Circuit.h(anc_qbit[0])
    Circuit.mcx(kq_qbits[:]+iq_qbits[:]+q_qbits[:] , anc_qbit[0])
    Circuit.h(anc_qbit[0])
    
    for i in range(len(kq_qbits)):
        Circuit.x(kq_qbits[i])
    for i in range(len(iq_qbits)):
        Circuit.x(iq_qbits[i])
    for i in range(len(q_qbits)):
        Circuit.x(q_qbits[i])
    Circuit.x(anc_qbit[0])

def A_gate( Circuit , DiagonalParams , current_time , Omega , Delta_t , Gamma_list , RegisterIndices):
    W_gate( Circuit , current_time , DiagonalParams , Delta_t , Omega , Gamma_list , RegisterIndices , False )
    R_gate( Circuit , RegisterIndices )
    W_gate( Circuit , current_time , DiagonalParams , Delta_t , Omega , Gamma_list , RegisterIndices , True )
    R_gate( Circuit , RegisterIndices )
    W_gate( Circuit , current_time , DiagonalParams , Delta_t , Omega , Gamma_list , RegisterIndices , False )

def Prepare_full_unitary_wo_Rgate( Circuit , DiagonalParams , Omega , Delta_t , Gamma_list , RegisterIndices , NumberOfSteps ):
    [kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , anc_qbits_idx] = RegisterIndices
    z_qbits = Circuit.qregs[z_qbits_idx]

    for ri in range( NumberOfSteps ):
        W_gate( Circuit , ri*Delta_t , DiagonalParams , Delta_t , Omega , Gamma_list , RegisterIndices , False )
        Circuit.append( diagonal_unitary(DiagonalParams, Delta_t) , z_qbits )

def Prepare_full_unitary( Circuit , DiagonalParams , Omega , Delta_t , Gamma_list , RegisterIndices , NumberOfSteps ):
    [kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , anc_qbits_idx] = RegisterIndices
    z_qbits = Circuit.qregs[z_qbits_idx]

    for ri in range( NumberOfSteps ):
        A_gate( Circuit , DiagonalParams , ri*Delta_t , Omega , Delta_t , Gamma_list , RegisterIndices)
        Circuit.append( diagonal_unitary(DiagonalParams, Delta_t) , z_qbits )
# =================================================================================== #
# ------------------------ State initialization and readouts ------------------------ #

def make_initial_state(init_zstate , number_of_spins , Q_max , M , K):
    Q = Q_max
    if M < 2:
        Niq = Q
    else:
        Niq = int(np.ceil(np.log2(M))) * Q

    Nkq = int(np.ceil(np.log2(K))) * Q

    psi_init_z = Statevector(init_zstate)
    psi_init_z = psi_init_z / np.linalg.norm(psi_init_z)

    total_circ_state = Statevector.from_label( '0' * Q)
    #total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * 2 ) ) # Two ancillas !
    #total_circ_state = total_circ_state.tensor( psi_init_z )
    total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Niq ) )
    total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Nkq ) )

    return total_circ_state