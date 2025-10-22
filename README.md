# MSc_thesis
 In this repository is present the code I used for my MSc thesis titled: "Entangled dynamical systems on networks: a density matrix approach"
## This code serves to simulate two networks undergoing Kuramoto dynamics and subjected to the same external field. I take into account two different situations:
- Simple case: the dynamics of the two network is independent, the dynamical equations are $\dot{\theta_i} = \omega_i + c \sum_{j = 1}^N A_{ij} \sin(\theta_j - \theta_i) + b\ \sin(\omega_f t - \theta_i) \quad i = 1,..,N$
- Complex case: the dynamics of the two networks is correlated: the interedependence is inserted in the external coupling that depends on the state of the two networks

Concerning the structure of the code, the folder *examples* contains the jupyter files to perform the main plots and some calculations:
- *biglyap* contains the calculation of the Lyapunov exponents for different parameter values
The folder "kuramoto" contains the python files to perform the majority of the calculations
