## Simluation of Floquet systems using quantum circuits

In this code, we simulate floquet systems exhibiting exotic phases of matter using linera combination of unitaries (LCU) method. 
The simulation consists of two parts:

1- The diagonal unitary $U_0(\Delta t)$

2- The off-diagonal untiary $U_{od}(\Delta t)$

Using these two components, we compose the entire unitary $U(t) \approx U_0(\Delta t) U_{od}(\Delta t) \ldots U_0(\Delta t) U_{od}(\Delta t)$ , where there are $r = \lfloor t / \Delta t \rfloor$ number of repeated
$U_0(\Delta t) U_{od}(\Delta t)$ terms.


### Issues, tests, and debugging

The off-diagonal unitary $U_{od}(\Delta t) = - WRW^\dagger R W$ consists of the circuits $W$ and $R$. $W = B^\dagger U_c B$, where $B$ is a state preparation gate and $U_c$ is a controlled unitary. 
The following tests have successfully been carried out

1- $B^\dagger B = B \, B^\dagger = 1$

2- $W^\dagger W = W  W^\dagger = 1$

3- $B$ produces the correct state for $n=2$, $Q = 2 , 3$.

We have also verified that the $W$ gate correctly simulates our desired $U = \sum_{i_q , k_q} \beta_{i_q , k_q} V_{i_q , k_q}$. Recall that $W |\psi\rangle = \frac{1}{s} U_{od} |\psi\rangle + |\Phi\rangle$. It is important to note that we have **failed** to verify that $W$ simulates the off-diagonal part of the *exact* unitary $U(t)$.

#### To do

1- Plot the overlap squared of the post-selected state after simulating $U_0 W$ (state denoted by $\psi_{ap}(t)$) and the exactly simulted state using mathematica denoted by $\psi_e(t)$.  
