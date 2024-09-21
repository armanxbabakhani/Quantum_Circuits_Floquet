import qiskit as qk
import numpy as np
import scipy.linalg as lin
import math
from qiskit.quantum_info import Statevector , Operator , partial_trace , DensityMatrix
from qiskit.circuit.library.standard_gates import HGate , XGate
from qiskit.circuit.library import GlobalPhaseGate
from qiskit.extensions import Initialize , UnitaryGate
from qiskit.providers.aer import AerSimulator
import sys
import matplotlib.pyplot as plt


# ===================== Quantum Operators =================== #
X = np.array([[0 , 1],[1 , 0]])
X2 = np.kron(np.eye(2) , X)
X1 = np.kron(X , np.eye(2))

Z = np.array([[1 , 0],[0 , -1]])
Z2 = np.kron(np.eye(2) , Z)
Z1 = np.kron(Z , np.eye(2))

#====================== Circuit Paramters =================== #

n = 2 # number of particles (qubits)
OffD_number = n-1 # number of permutations
M = OffD_number
C0 = 1.0 # Parameter for the floquet interaction strength
Coeffs = C0*np.array([2+(-1)**x for x in np.arange(1,n)]) # there are n elements in this array
Omega = 1.0
h0 = 1.0 * 0.01
Longit_h = h0*np.array([1.5 + 0.5*(-1)**x for x in range(n)])
Vx = 1.0 * 0.01
p = 2 # number of periods
K = 2 # number of frequencies per permutation
C = np.max(Coeffs)

diags = (Omega/2)*Coeffs
Gammas = diags
Gamma = OffD_number * Omega * C0 * K #  Gamma = M K max_i Gamma_i = (n-1) \omega c
#Gamma = OffD_number * Omega * 0.5
Delta_t = np.log(2)/Gamma


# ------------------ Taking the final evolution time as an input argument --------------------- #
final_time = float(sys.argv[1])

t = 0
time = []
X1_data = []
X2_data = []
Z1_data = []
Z2_data = []


psi_approx = []

#math.ceil(np.log(r/eps)/np.log(np.log(r/eps)))
#print('In the precomuting stage, Q is {Q}')
#if Q > 2:
#    Q = 2

def Gammaf(iq):
        G = (2 + (-1)**iq[0])
        q = len(iq)
        for j in np.arange(1,q):
            G = G*(2 + (-1)**iq[j])
        return G*(Omega * C0 / 2)**q

Gam = np.sum([Gammaf([x]) for x in np.arange(1, n)])
if Gam == 0:
    GDt = 0.0
else:
    Delta_t = np.log(2)/Gam
    GDt = Gam*Delta_t

t += Delta_t
while t < final_time:
    
    r = int(t/Delta_t)

    eps = 0.05 # 5% tolerance
    print(f't is {t} and r is {r}')
    Q = 2
# ============================= Constructing the unitary U_0 =============================== #
# ---------------------------------- Diagonal Part (H_0) ----------------------------------- #
    #def Diagonal_U0_old(hs , Vs , time):
    #    U0 = qk.QuantumCircuit(n+1)
    #    # For single Z rotations (with h_i):
    #    for x in range(n):
    #        U0.cx(x , n)
    #        U0.rz(-hs[x]*time , n)
    #        U0.cx(x , n)

        # For two-body ZZ interactions (with V_x):
    #    for x in range(n-1):
    #        U0.cx(x , n)
    #        U0.cx(x+1 , n)
    #        U0.rz(-Vs*time , n)
    #        U0.cx(x , n)
    #        U0.cx(x+1 , n)
    #    return U0.to_gate() 
    
    def Diagonal_U0(hs , Vs , ev_time):
        U0 = qk.QuantumCircuit(n + 1)

        # For single Z rotations (with h_i):
        for x in range(n):
            U0.rz(-2*hs[x]*ev_time , x)

        # For two-body ZZ interactions (with V_x):
        for x in range(n-1):
            U0.cx(x , x+1)
            U0.rz(-2*Vs*ev_time , x+1)
            U0.cx(x , x+1)
        return U0.to_gate()
    
    def Diagonal_U0_circ(Circuit , z_qbits_index,  hs , Vs , time):
        z_qbits = Circuit.qregs[z_qbits_index]
        # For single Z rotations (with h_i):
        for x in range(n):
            Circuit.cx(z_qbits[x] , z_qbits[n])
        Circuit.rz(2*hs[x]*time , z_qbits[n])
        for x in range(n):
            Circuit.cx(z_qbits[x] , z_qbits[n])

        # For two-body ZZ interactions (with V_x):
        for x in range(n-1):
            Circuit.cx(z_qbits[x] , z_qbits[n])
            Circuit.cx(z_qbits[x + 1] , z_qbits[n])
        Circuit.rz(2*Vs*time , z_qbits[n])
        for x in range(n-1):
            Circuit.cx(z_qbits[x] , z_qbits[n])
            Circuit.cx(z_qbits[x + 1] , z_qbits[n])

    # ---------------- The REFLECTION (R) about the zero state of Q+1 registers -----------
    # Note: We may have to do this on (Q+1)*log2(M) registers instead to
    #      make things for convenient to use

    def R_gate(Circuit , kq_qbits_idx , iq_qbits_idx , anc_qbit_idx , q_qbits_idx):
        #simulator = AerSimulator()
        #transcirc = qk.transpile(Circuit , simulator)
        #l0 = len(transcirc.count_ops())
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
        # Useful functions:

    # Gammaf computes Gamma_i_q! if i_q is a vector it will return the multiplication of all the sub-gammas

    #GDt = 0.05

    # These two functions are used if we manually prepare the state without state preparation call from qiskit built-in functions
    # For now, we will use qiskit built-in functions (so these functions are not used!)
    def Q_angle(Qq , i):
        angle = 0.0
        for q in np.arange(1 , Qq - i+1):
            angle += GDt**(q) / math.factorial(q + i)
        angle *= math.factorial(i)
        angle = np.sqrt(angle)

        return np.arctan(angle)

    def UQ_circuit(Qq , n , dagger=False):
        L = int(np.log2(n))
        uq = qk.QuantumCircuit(Qq*L)
        if dagger:
            for q in range(Qq-1 , 0 , -1):
                uq.cry(-2.0*Q_angle(Qq , q) , (q-1)*L , q*L)
            uq.ry(-2.0*Q_angle(Qq, 0) , 0)  
        else:
            uq.ry(2.0*Q_angle(Qq, 0) , 0)
            for q in np.arange(1,Qq):
                uq.cry(2.0*Q_angle(Qq , q) , (q-1)*L , q*L) 
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

    # ================================= The controlled unitary Uc_P =============================== #
    # Adopting the convention of labels: if P_i represents X_i X_i+1, e.g. |i=5> corresponds to X_5 X_6 permutation

    # To build a generalized controlled unitary operator, we need to specify a set of controlled qubits (q_c),
    #    and act upon these controlled unitary qubits with a sequence of controlled permutations X_i X_i+1 s

    # Defining a controlled XX gate
    L = int( np.log2(n) )
    # Defining the function Uc_P to produce controlled unitary gates on a set of control qubits and target qubits!

    # ctr_qbits_idx will be the index for the set of quantum registers of the Circuit
    #    that act as a control qubit.. Same for targ_qbits_idx

    # Note that iq_qbits register has Q*log2(n) number of qubits:
    #   The i_sub_idx is a variable to specify the permutation for the specific sub index of the iq_qbits

    def Uc_P_old( Circuit , ctr_qbits_idx , targ_qbits_idx , i_sub_idx ):
        # Assuming i_sub_idx refers to a specific block of register of size log2(n) qubits
        if i_sub_idx > 0:
            ctr_qbits = Circuit.qregs[ctr_qbits_idx]
            targ_qbits = Circuit.qregs[targ_qbits_idx]
            L = int(np.log2(n))
            N_qbits = n + Q * L # Total qubits = z qubits + i_q qubits + k_q qubits
            
            ctr_qbits_subset = ctr_qbits[0:i_sub_idx*L]
            Lctrs = len( ctr_qbits_subset )
            
            # Making the controlled XX gate for the specific number of controlled qubits
            cxxcirc = qk.QuantumCircuit(2 , name='cXX')
            cxxcirc.x( range(2) )
            cxxGate = cxxcirc.to_gate()
            cxxGate = cxxGate.control(Lctrs)
            
            for i in np.arange(1,n):
                for k in range(i_sub_idx):
                    ibin = bin(i + k*n)[2:]
                    ibin = (Lctrs-len(ibin))*'0' +ibin 
                    zer_ctrs = [x for x in range(Lctrs) if(int(ibin[Lctrs-1 - x]) == 0)]
                    for j in range(len(zer_ctrs)):
                        Circuit.x(ctr_qbits[zer_ctrs[j]])
                    # This part needs to be fixed.. We need to 
                    Circuit.append( cxxGate , ctr_qbits[0:Lctrs] + targ_qbits[i-1:i+1] )
                    for j in range(len(zer_ctrs)):
                        Circuit.x( ctr_qbits[zer_ctrs[j]] )

    def Uc_P( Circuit , ctr_qbits_idx , targ_qbits_idx , i_sub_idx , dagger):
    # Assuming i_sub_idx refers to a specific block of register of size log2(n) qubits
        if i_sub_idx > 0:
            ctr_qbits = Circuit.qregs[ctr_qbits_idx]
            targ_qbits = Circuit.qregs[targ_qbits_idx]
            L = int( np.log2(n) )
            N_qbits = n + Q * L # Total qubits = z qubits + i_q qubits + k_q qubits
            
            ctr_qbits_subset = ctr_qbits[(i_sub_idx-1)*L:i_sub_idx*L]
            # Making the controlled XX gate for the specific number of controlled qubits
            cxxcirc = qk.QuantumCircuit(2 , name='cXX')
            cxxcirc.x( range(2) )
            cxxGate = cxxcirc.to_gate()
            cxxGate = cxxGate.control(L)
            
            if dagger:
                for i in np.arange(n-1 , 0 , -1):
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

    # =============================== The controlled unitary Uc_Phi =================================== #
    # UC_Phi_Omega(k_qbits) generates the omega related phases on the k_q registers
    # The only thing that needs to be tested in the Uc_Phi_Omega!

    def Uc_Phi_Omega(Circuit , k_qbits_idx , q_qbits_idx , dagger= False):
        # Apply i\omega \sum_l=j^q (-1)^k_l
        # Omega is a global variable
        k_qbits = Circuit.qregs[k_qbits_idx]
        q_qbits = Circuit.qregs[q_qbits_idx]
        
        delta = 2*Omega*Delta_t # Dividing the angle by 2 for rz

        if dagger:
            for l in range(len(k_qbits)-1 , - 1 , -1):
                if l > 0:
                    for lk in range(l-1 , -1 , -1):
                        Circuit.crz( delta/((l+2)*(l+1)) , q_qbits[l] , k_qbits[lk])
                Circuit.crz( -(l+1)*delta/(l+2) , q_qbits[l]  , k_qbits[l])
        else:
            for l in range(len(k_qbits)):
                Circuit.crz((l+1)*delta/(l+2) , q_qbits[l] , k_qbits[l])
                if l > 0:
                    for lk in range(l):
                        Circuit.crz(-delta/((l+2)*(l+1)) , q_qbits[l] , k_qbits[lk])

    def U0_q_ctrl( Circuit , delta_t , q_qbits , z_qbits , dagger=False):
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
        iq_qbits = Circuit.qregs[iq_qbits_idx]
        q_qbits = Circuit.qregs[q_qbits_idx]
        z_qbits = Circuit.qregs[z_qbits_idx]
        liq = iq_qbits.size
        if dagger:
            # Circuit.append( Diagonal_U0( Longit_h , Vx , Delta_t ) , z_qbits )
            U0_q_ctrl( Circuit , -Delta_t*Q , q_qbits , z_qbits , False)
            for j in range(Q , 0 , -1):
                Uc_P( Circuit , iq_qbits_idx , z_qbits_idx , j , dagger)
                U0_q_ctrl( Circuit , Delta_t , q_qbits , z_qbits , False )
        else:
            for j in np.arange(1,Q+1):
                U0_q_ctrl( Circuit , Delta_t , q_qbits , z_qbits , True )
                Uc_P( Circuit , iq_qbits_idx , z_qbits_idx , j , dagger)
            U0_q_ctrl( Circuit , -Delta_t*Q , q_qbits , z_qbits , True )


    def B_prepare(Circuit , kq_qbits_idx , iq_qbits_idx , q_qbits_idx ,  dagger=False):
        mapping_vec = np.zeros((1 , n-1))
        for i in np.arange(1,n):
            mapping_vec[0][i-1] = np.sqrt( Gammaf([i]) )
        mapping_vec[0] = mapping_vec[0]/np.sqrt( np.dot(mapping_vec[0],mapping_vec[0]) )
            
        L = int( np.log2(n) )
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
        
    def W_gate(Circuit , kq_qbits_idx , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger = False):
        B_prepare( Circuit , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , False )
        Uc_Phi( Circuit , iq_qbits_idx , z_qbits_idx , q_qbits_idx , dagger )
        Uc_Phi_Omega( Circuit , kq_qbits_idx , q_qbits_idx , dagger )
        B_prepare( Circuit , kq_qbits_idx , iq_qbits_idx , q_qbits_idx , True )

    def A_gate( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index ):
        anc_qubits_index = 3
        W_gate( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index , False )
        #R_gate( Circuit , kq_qbits_index , iq_qbits_index , anc_qubits_index , q_qbit_index )
        #W_gate( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index , True )
        #R_gate( Circuit , kq_qbits_index , iq_qbits_index , anc_qubits_index , q_qbit_index )
        #W_gate( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index , False )

    def Prepare_full_unitary( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index , r_number ):
        kq_qbits = Circuit.qregs[kq_qbits_index]
        iq_qbits = Circuit.qregs[iq_qbits_index]
        z_qbits = Circuit.qregs[z_qbits_index]
        print( f'time is {t}' )
        # print( f'the r number is: {r_number}' )
        for ri in range( r_number ):
            Circuit.append( Diagonal_U0(Longit_h , Vx , -Delta_t) , z_qbits )
            A_gate( Circuit , kq_qbits_index , iq_qbits_index , z_qbits_index , q_qbit_index )
            #W_gate( Circuit , 0 , 1 , 2 , 4 , False )
        # print( f'Circuit is prepared and ready for simulation' )

    def Make_initial_state_old(init_zstate):    
        Nkq = int(np.log2(K)) * Q
        Niq = int(np.log2(n)) * Q

        psi_init_z = Statevector(init_zstate)
        psi_init_z = psi_init_z / np.linalg.norm(psi_init_z)

        total_circ_state = Statevector.from_label( '0' * Nkq )
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Niq ) )
        total_circ_state = total_circ_state.tensor( psi_init_z )
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * 2 ) ) # Two ancillas !
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Q ) ) # The q qubits!

        return total_circ_state

    def Make_initial_state(init_zstate):
        Nkq = int(np.log2(K)) * Q
        Niq = int(np.log2(n)) * Q

        psi_init_z = Statevector(init_zstate)
        psi_init_z = psi_init_z / np.linalg.norm(psi_init_z)

        total_circ_state = Statevector.from_label( '0' * Q)
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * 2 ) ) # Two ancillas !
        total_circ_state = total_circ_state.tensor( psi_init_z )
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Niq ) )
        total_circ_state = total_circ_state.tensor( Statevector.from_label( '0' * Nkq ) )

        return total_circ_state

    def Obtain_zstate_density(total_circ_state):
        Nkq = int(np.log2(K))*Q
        Niq = int(np.log2(n))*Q
        Ntotal = Nkq + Niq + n + 2 + Q
        
        traced_qubits = list(range(Nkq + Niq)) + list(np.arange(Ntotal-1 , Ntotal-Q-3 , -1))
        final_state_density = partial_trace(DensityMatrix(total_circ_state), traced_qubits)

        return final_state_density


    # ==================================== THE SIMULATION ====================================== #
    # ---------------------------- Initializing the quantum circuit ---------------------------- #
    Nkq = int(np.log2(K))*Q
    Niq = int(np.log2(n))*Q
    Ntotal = Nkq + Niq + n + 2 + Q

    kqQubits = qk.QuantumRegister(Nkq , '|kq>')
    iqQubits = qk.QuantumRegister(Niq , '|iq>')
    zQubits = qk.QuantumRegister(n + 1 , '|z>') # n+1 st is an ancilla for diagonal rotations!
    ancQubit = qk.QuantumRegister(1 , '|anc>')
    qQubits = qk.QuantumRegister(Q , '|q>')

    kq_qbits_index = 0
    iq_qbits_index = 1
    z_qbits_index = 2
    anc_qbits_index = 3
    q_qbits_index = 4

    FullCirc = qk.QuantumCircuit( kqQubits , iqQubits , zQubits , ancQubit , qQubits )


    # =========================== Building the circuits  U_0(dt) U_od(dt) ... U_0(dt) U_od(dt) =================================== #
    FullCirc = qk.QuantumCircuit( kqQubits , iqQubits , zQubits , ancQubit, qQubits )
    Prepare_full_unitary( FullCirc , kq_qbits_index, iq_qbits_index , z_qbits_index , q_qbits_index , r )

    # --------------------------- Initializing the state \psi(0) ----------------------------- #
    state = [1.0 , 0.0 , 0.0 , 0.0]
    state_norm = state/np.linalg.norm( state )
    total_circuit_state = Make_initial_state(state)
    #total_circuit_state = Statevector.from_label( '0'*( 3 * Q + n + 2 ) )

    # Evolve the initialized state
    final_total_state = total_circuit_state.evolve( FullCirc )
    # final_state_density = np.array( Obtain_zstate_density( final_total_state ) )

    # ---------------- Picking out the |z> state (tracing out the ancillas) ------------------ #
    increment = 2**( 2*Q )
    final_state = np.array( [final_total_state[0] , final_total_state[increment] , final_total_state[2*increment] , final_total_state[3*increment]] )
    final_state = final_state/np.linalg.norm( final_state )
    print( 'final state is: ' , final_state )
    psi_approx.append(final_state)

    final_state_conj = np.conjugate( final_state )
    final_density = np.outer( final_state , final_state_conj )

    X1_data.append(np.trace(np.dot(final_density , X1)))
    X2_data.append(np.trace(np.dot(final_density , X2)))
    Z1_data.append(np.trace(np.dot(final_density , Z1)))
    Z2_data.append(np.trace(np.dot(final_density , Z2)))

    time.append(t)
    t += Delta_t


print(f'The simulation is done for Q = {Q} , n = {n} , Delta t = {Delta_t} , GammaDt = {GDt} , Omega = {Omega} , Vx = {Vx}')

# ============================ Comparing to the exact results =======================================

psi_exact_0 = [[0.823604 - 0.268856j, 0 , 0 , 0.413975 + 0.279316j] ,
             [0.760281 - 0.250455j, 0 , 0 , -0.564344 - 0.201893j] , 
             [0.591217 - 0.563383j, 0. , 0. , -0.308169 + 0.487946j] ,
             [0.761383 - 0.249421j, 0., 0., 0.359156 - 0.478634j] , 
             [0.438596 - 0.841627j, 0, 0, -0.207798 - 0.236888j] , 
             [0.6971 - 0.656576j, 0, 0, 0.245904 + 0.149966j] , 
             [0.0394336 - 0.954782j, 0, 0, -0.27888 + 0.0951996j] , 
             [0.202655 - 0.939304j, 0, 0, 0.0554213 + 0.27123j] , 
             [-0.0238209 - 0.722283j, 0 , 0 , 0.0690068 - 0.687734j] , 
             [-0.324552 - 0.899476j, 0 , 0 , -0.288344 + 0.0496678j]]

psi_exact_small = [[-0.981414 + 0.0167519j, 0, 0, 0.00383924 - 0.19113j] ,
             [0.469779 + 0.00467553j, 0, 0, -0.0452865 - 0.881609j] , 
             [0-0.830634 + 0.0314374j, 0, 0, 0.0420445 + 0.554338j] ,
             [-0.430186 + 0.045801j, 0, 0, 0.0768779 + 0.898293j] , 
             [-0.238795 + 0.0130824j, 0, 0, -0.0938423 - 0.966436j] , 
             [-0.888583 + 0.123591j, 0, 0, -0.0440975 - 0.439546j] , 
             [0.668084 - 0.114349j, 0, 0, 0.0906796 + 0.729631j] , 
             [-0.974912 + 0.159473j, 0, 0, 0.032855 + 0.151773j] , 
             [0.94897 - 0.171533j, 0, 0, 0.0478432 + 0.260265j] , 
             [-0.949677 + 0.188929j, 0, 0, -0.0428947 - 0.246128j]]

psi_exact = psi_exact_small

#psi_approx = psi_approx[1::]
if(len(psi_exact) == len(psi_approx)):
    overlaps = []
    for i in range(len(psi_exact)):
        overlaps.append(np.abs(np.dot(np.array(psi_approx[i]) , np.array(psi_exact[i])))**2)

    plt.figure(figsize=(10, 6))
    plt.plot(time, overlaps, label='overlap', color='blue' , marker='x')
    plt.title('The overlaps of the approximate simulation and the exact')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.show()
else:
    print('Comparison plot cannot be made: the number of points in the exact wavefunction do not match the number of simulated points! ')

# ================== Plotting the expectation value of operators ==================== #
plt.figure(figsize=(10, 6))
plt.plot(time, X1_data, label='X1', color='blue' , marker='x')
plt.plot(time, X2_data, label='X2', color='red' )
plt.plot(time, Z1_data, label='Z1', color='black' , marker='o')
plt.plot(time, Z2_data, label='Z2', color='green')
plt.title('Simulation of the Diagonal part')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.legend()
plt.show()
