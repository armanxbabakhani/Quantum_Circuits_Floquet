import qiskit as qk
import numpy as np
import scipy.linalg as lin
import math
from qiskit.quantum_info import Statevector , Operator , partial_trace , DensityMatrix
from qiskit.circuit.library.standard_gates import HGate , XGate
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.extensions import Initialize , UnitaryGate
from qiskit.providers.aer import AerSimulator



def Diagonal_U0(hs , Vs , ev_time):
    """
    
    Input #1 (hs): an array f doubles hopping strengths 
    Input #2 (Vs): a coupling strength V for all the XX interactions
    Input #3 (ev_time): the time of evolution

    output: a unitary gate simulating the diagonal evolution U_0(t = ev_time)

    """

    U0 = qk.QuantumCircuit(n + 1)

    for x in range(n):
        U0.rz(2*hs[x]*ev_time , x)

    # For two-body ZZ interactions (with V_x):
    for x in range(n-1):
        U0.cx(x , x+1)
        U0.rz(2*Vs*ev_time , x+1)
        U0.cx(x , x+1)
    return U0.to_gate()

# =================================================================================== #
# ------------------------------ LCU State preparation ------------------------------ #

def Q_angle(Q_max , GDt , i):
    angle = 0.0
    for q in np.arange(1 , Q_max - i+1):
        angle += GDt**(q) / math.factorial(q + i)
    angle *= math.factorial(i)
    angle = np.sqrt(angle)

    return np.arctan(angle)


def UQ_circuit(Q_max , number_of_spins , dagger=False):
    L = int(np.log2(number_of_spins))
    uq = qk.QuantumCircuit(Q_max*L)
    if dagger:
        for q in range(Q_max-1 , 0 , -1):
            uq.cry(-2.0*Q_angle(Q_max , q) , (q-1)*L , q*L)
        uq.ry(-2.0*Q_angle(Q_max , 0) , 0)  
    else:
        uq.ry(2.0*Q_angle(Q_max , 0) , 0)
        for q in np.arange(1,Q_max):
            uq.cry(2.0*Q_angle(Q_max , q) , (q-1)*L , q*L) 
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

def UnitMap(vec , dagger=False):
    N = vec[0].size + 1
    U = np.zeros((N,N))
    U[0][0] = 1.0
    unit = create_unitary(vec)
    U[1: , 1:] = unit
    if dagger:
        U = U.conj().T
    return UnitaryGate(U)


def B_prepare(Circuit , number_of_spins , Q_max , Gamma_function , kq_qbits_idx , iq_qbits_idx , q_qbits_idx ,  dagger=False):
        
    """
    This is a void function preparing the states on |i_q> , |k_q> , and |q> registers.

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (int) #1 (number_of_spins): The number of spins of the system
    Input (int) #2 (Q_max): The maximum expansion order Q
    Input (int) #3 (Gamma_function): The function specifying the off-diagonal coefficients of the Hamiltonian
    Input (int) #4 (kq_qbits_idx): The index of the register of the |k_q> qubits
    Input (int) #5 (iq_qbits_idx): The index of the register of the |i_q> qubits
    Input (int) #3 (q_qbits_idx): The index of the register of the |q> qubits
    Input (bool) #7 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 
    
    """
    n = number_of_spins
    L = int( np.log2(n) )
    Q = Q_max
    mapping_vec = np.zeros((1 , n-1))

    for i in np.arange(1,n):
        mapping_vec[0][i-1] = np.sqrt( Gamma_function([i]) )
    mapping_vec[0] = mapping_vec[0]/np.sqrt( np.dot(mapping_vec[0],mapping_vec[0]) )
        
    iq_qbits = Circuit.qregs[iq_qbits_idx]
    kq_qbits = Circuit.qregs[kq_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]

    Umapdag = UnitMap(mapping_vec , True)
    Umap = UnitMap(mapping_vec , False)
    
    if dagger:
        for i in range(Q-1 , -1 , -1):
            Circuit.append(Umapdag , iq_qbits[list(np.arange(i*L , (i+1)*L))])
        for i in range(Q-1 , -1 , -1):
            Circuit.ch(iq_qbits[i*L] , kq_qbits[i])
        for i in range(Q-1 , -1 , -1):
            Circuit.cx(iq_qbits[i*L] , q_qbits[i])
        Circuit.append(UQ_circuit(Q , n , dagger) , iq_qbits)

    else:
        Circuit.append(UQ_circuit(Q , n , dagger) , iq_qbits)
        for i in range(Q):
            Circuit.ch(iq_qbits[i*L] , kq_qbits[i])
        for i in range(Q):
            Circuit.cx(iq_qbits[i*L] , q_qbits[i])
        for i in range(Q):
            Circuit.append(Umap , iq_qbits[list(np.arange(i*L , (i+1)*L))])


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
#   The i_sub_idx is a variable to specify the permutation for the specific sub index of the iq_qbits

def Uc_P( Circuit , number_of_spins , ctr_qbits_idx , targ_qbits_idx , i_sub_idx , dagger):
    
    """"
    This is a void function creating a controlled permutation on the circuit

    Input (int) #1 (ctr_qbits_idx): The index of the register for the controlled qubits
    Input (int) #2 (targ_qbits_idx): The index of the register for the target qubits
    Input (int) #3 (i_sub_idx): The sub-index (q) for the i_q register to generate the controlled qubit
    Input (bool) #4 (dagger): Speficies whether the gate is hermitian conjugate of U_p or not 

    """

    # Assuming i_sub_idx refers to a specific block of register of size log2(n) qubits
    if i_sub_idx > 0:
        ctr_qbits = Circuit.qregs[ctr_qbits_idx]
        targ_qbits = Circuit.qregs[targ_qbits_idx]
        L = int( np.log2(number_of_spins) )
        N_qbits = Circuit.num_qubits # Total qubits for the circuit
        
        ctr_qbits_subset = ctr_qbits[(i_sub_idx-1)*L:i_sub_idx*L]
        # Making the controlled XX gate for the specific number of controlled qubits
        cxxcirc = qk.QuantumCircuit(2 , name='cXX')
        cxxcirc.x( range(2) )
        cxxGate = cxxcirc.to_gate()
        cxxGate = cxxGate.control(L)
        
        if dagger:
            for i in np.arange(number_of_spins-1 , 0 , -1):
                ibin = bin(i)[2:]
                zer_ctrs = [x for x in range(L) if(int(ibin[x]) == 0)]
                for j in range(len(zer_ctrs)):
                    Circuit.x( ctr_qbits_subset[zer_ctrs[j]] )
                Circuit.append( cxxGate , ctr_qbits_subset[:] + targ_qbits[i-1:i+1] )
                for j in np.arange(len(zer_ctrs)-1 , -1 , -1):
                    Circuit.x( ctr_qbits_subset[zer_ctrs[j]] )
        else:
            for i in np.arange(1,n):
                ibin = bin(i)[2:]
                #ibin = (Lctrs-len(ibin))*'0' +ibin 
                zer_ctrs = [x for x in range(L) if(int(ibin[x]) == 0)]
                for j in range(len(zer_ctrs)):
                    Circuit.x( ctr_qbits_subset[zer_ctrs[j]] )
                # This part needs to be fixed.. We need to 
                Circuit.append( cxxGate , ctr_qbits_subset[:] + targ_qbits[i-1:i+1] )
                for j in range(len(zer_ctrs)):
                    Circuit.x( ctr_qbits_subset[zer_ctrs[j]] )


def U0_q_ctrl( Circuit , delta_t , q_qbits , z_qbits , dagger=False):
    """"
    This is a void function creating the q-dependent controlled diagonal rotations 

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (double) #1 (delta_t): Specifying the rotation amount, for which a U_0( delta_t/(q+1) ) will be applied 
    Input (int) #2 (q_qbits): The quantum register (not the index) of the q qubits
    Input (int) #3 (z_qbits): The quantum register of the z qubits
    Input (bool) #4 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 

    """
    if dagger:
        for k in range(len(q_qbits)-1 , 0 , -1):
            CUk = Diagonal_U0( Longit_h , Vx , delta_t/((k+1)*(k+2)) )
            CUk = CUk.control()
            Circuit.append( CUk , [q_qbits[k]] + z_qbits[:] )
        CU0 = Diagonal_U0( Longit_h , Vx , -delta_t/(2.0) )
        CU0 = CU0.control()
        Circuit.append( CU0 , [q_qbits[0]] + z_qbits[:] )
    else:
        CU0 = Diagonal_U0( Longit_h , Vx , delta_t/(2.0) )
        CU0 = CU0.control()
        Circuit.append( CU0 , [q_qbits[0]] + z_qbits[:] )
        for k in np.arange(1, len(q_qbits)):
            CUk = Diagonal_U0( Longit_h , Vx , -delta_t/((k+1)*(k+2)) )
            CUk = CUk.control()
            Circuit.append( CUk , [q_qbits[k]] + z_qbits[:] )

# UC_Phi(iq_qbits , z_qbits , q) generates the E_z_iq and E_z_ij related phases on |z>
def Uc_Phi(Circuit  , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger=False):
    """
    This is a void function creating the q-dependent controlled rotations due to the off-diagonal expansion.

    Input (QuantumCircuit) #0: The void variable which is a quantum circuit on which U_c\omega is appended
    Input (int) #1 (iq_qbits_idx): The index of the register of the |i_q> qubits
    Input (int) #2 (z_qbits_idx): The index of the register of the |z> qubits
    Input (int) #3 (q_qbits_idx): The index of the register of the |q> qubits
    Input (bool) #4 (dagger): Speficies whether the gate is hermitian conjugate of the gate or not 
    
    """

    iq_qbits = Circuit.qregs[iq_qbits_idx]
    q_qbits = Circuit.qregs[q_qbits_idx]
    z_qbits = Circuit.qregs[z_qbits_idx]
    liq = iq_qbits.size
    if dagger:
        #Circuit.append( Diagonal_U0( Longit_h , Vx , Delta_t ) , z_qbits )
        U0_q_ctrl( Circuit , -Delta_t*Q , q_qbits , z_qbits , True)
        for j in range(Q , 0 , -1):
            Uc_P( Circuit , iq_qbits_idx , z_qbits_idx , j , dagger )
            U0_q_ctrl( Circuit , Delta_t , q_qbits , z_qbits , True )

            Circuit.cp( np.pi , q_qbits[j-1] , z_qbits[0] )
            Circuit.crz( -np.pi , q_qbits[j-1] , z_qbits[0] )
    else:
        for j in np.arange(1,Q+1):
            Circuit.crz( np.pi , q_qbits[j-1] , z_qbits[0] )
            Circuit.cp( -np.pi , q_qbits[j-1] , z_qbits[0] )

            U0_q_ctrl( Circuit , Delta_t , q_qbits , z_qbits , False )
            Uc_P( Circuit , iq_qbits_idx , z_qbits_idx , j , dagger )
        U0_q_ctrl( Circuit , -Delta_t*Q , q_qbits , z_qbits , False )


# =================================================================================== #
# ---------------------- Generating the off-diagonal unitary ------------------------ #

def W_gate(Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger = False):
    B_prepare( Circuit , number_of_spins , Q_max , Gamma_function , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , False )
    Uc_Phi( Circuit , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger )
    Uc_Phi_Omega( Circuit , current_time , kq_qbits_idx , q_qbits_idx , dagger )
    B_prepare( Circuit , number_of_spins , Q_max , Gamma_function , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , True )

# =================================================================================== #
# ------------------------ Amplitude Amplification functions ------------------------ #

# ---------------- The REFLECTION (R) about the zero state of Q+1 registers -----------
# Note: We may have to do this on (Q+1)*log2(M) registers instead to
#      make things for convenient to use

def R_gate(Circuit , kq_qbits_idx , iq_qbits_idx , anc_qbit_idx , q_qbits_idx):

    kq_qbits = Circuit.qregs[kq_qbits_idx]
    iq_qbits = Circuit.qregs[iq_qbits_idx]
    anc_qbit = Circuit.qregs[anc_qbit_idx]
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

def A_gate( Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index ):
    anc_qubits_index = 3
    W_gate(Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbits_index , False)
    R_gate( Circuit , kq_qbits_index , iq_qbits_index , anc_qubits_index , q_qbit_index )
    W_gate(Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbits_index , True)
    R_gate( Circuit , kq_qbits_index , iq_qbits_index , anc_qubits_index , q_qbit_index )
    W_gate(Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbits_index , False)

def Prepare_full_unitary( Circuit , number_of_spins , Q_max , current_time , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index , r_number ):
    kq_qbits = Circuit.qregs[kq_qbits_index]
    iq_qbits = Circuit.qregs[iq_qbits_index]
    z_qbits = Circuit.qregs[z_qbits_index]

    for ri in range( r_number ):
        A_gate( Circuit , number_of_spins , Q_max , ri*Delta_t , Gamma_function , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index )
        Circuit.append( Diagonal_U0(Longit_h , Vx , Delta_t) , z_qbits )


# =================================================================================== #
# ------------------------ State initialization and readouts ------------------------ #

def Make_initial_state(init_zstate , number_of_spins , Q_max):
    K = 2 # Only two exponential diagonal entries for cos(omega t)
    Nkq = int(np.log2(K)) * Q
    Niq = int(np.log2(number_of_spins)) * Q

    psi_init_z = Statevector(init_zstate)
    psi_init_z = psi_init_z / np.linalg.norm(psi_init_z)

    total_circ_state = Statevector.from_label( '0' * Q)
    total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * 2 ) ) # Two ancillas !
    total_circ_state = total_circ_state.tensor( psi_init_z )
    total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Niq ) )
    total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Nkq ) )

    return total_circ_state