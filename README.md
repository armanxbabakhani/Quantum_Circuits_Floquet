## Simluation of Floquet systems using quantum circuits

In this code, we simulate floquet systems exhibiting exotic phases of matter using linera combination of unitaries (LCU) method. 
The simulation consists of two parts:

1- The diagonal unitary $U_0(\Delta t)$

2- The off-diagonal untiary $U_{od}(\Delta t)$

Using these two components, we compose the entire unitary $U(t) \approx U_0(\Delta t) U_{od}(\Delta t) \ldots U_0(\Delta t) U_{od}(\Delta t)$ , where there are $r = floor(t / \Delta t)$ number of repeated
$U_0(\Delta t) U_{od}(\Delta t)$ terms.
