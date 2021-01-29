# One-Dimensional Neutron Transport Equation ($S_{N}$) Solver using the Discrete Ordinates Method

## Machine Learning Methods are used to reduce the data requirements of Scattering and Fission Cross-Sections

### Current Work
- [ ] Add Testing Functions with pytest
- [ ] Clean up $\texttt{source.py}$ and test functions
- [ ] Make time-dependent source problems for $\texttt{source.py}$
- [ ] Make $\texttt{class}$ to run through benchmark problems

### Current Data Saving Techniques
<!-- 1. $\texttt{correct.py}$ is the correct $S_{N}$ code for one-dimensional sweeps. -->
1. $\texttt{dj_prob.py}$ incorporates Deep Jointly-Informed Neural Networks<sup>1</sup> (DJINN) into the $S_{N}$ code for $\Sigma_\mathrm{s} \phi$ and $\Sigma_\mathrm{f} \phi$ calculations.
2. $\texttt{ae_prob.py}$ incorporates an autoencoder into the $S_{N}$ code for $\phi$, $\Sigma_\mathrm{s} \phi$, and $\Sigma_\mathrm{f} \phi$ to compress the energy groups and use in conjuction with DJINN. 
3. $\texttt{svd_prob.py}$ incorporates an SVD into the $S_{N}$ code for the $\Sigma_\mathrm{s}$ and $\Sigma_\mathrm{f}$ matrices.
4. $\texttt{hybrid.py}$ separates the collided and uncollided terms to be used with different numbers of ordinates ($N$) and energy groups ($G$). 

### Hybrid Method for Time Dependent Multigroup Problems
0. Initialize $\psi^n$ to zero
1. Calculate the uncollided $\psi^{n+1}_{u}$ and $\phi_{u}^{n+1}$ through the sweep
	\begin{equation}
	\Omega \cdot \nabla \psi_{u}^{n+1} + \left( \Sigma_\mathrm{t} + \frac{1}{v \Delta t} \right) \psi_{u}^{n+1} = Q_u + \frac{1}{v \Delta t} \psi_{u}^{n}
	\end{equation} \begin{equation} \begin{split}
	\psi_{u}^{n+1} \left( \frac{\mu_n}{\Delta x} + \frac{1}{2} \Sigma_\mathrm{t} + \frac{1}{2 v \Delta t} \right) &= \\ Q_u + \frac{1}{v \Delta t} \psi_{u}^{n} &+ \psi_{u}^{n+1} \left( \frac{\mu_n}{\Delta x} - \frac{1}{2} \Sigma_\mathrm{t} - \frac{1}{2 v \Delta t} \right)
	\end{split} \end{equation} 
2. Use $\phi_{u}^{n+1}$ to create source term ($Q_c$) for the collided equation: 
	\begin{equation} 
	Q_c = \Sigma_\mathrm{s} \phi_{u}^{n+1} + \Sigma_\mathrm{f} \phi_{u}^{n+1}
	\end{equation}
3. Solve the collided equation with the new source term ($Q_c$)
	\begin{equation}
	\Omega \cdot \nabla \psi_{c}^{n+1} + \left( \Sigma_\mathrm{t} + \frac{1}{v \Delta t} \right) \psi_{c}^{n+1} = \Sigma_\mathrm{s} \phi_{c}^{n+1} + \Sigma_\mathrm{f} \phi_{c}^{n+1} + Q_{c} 
	\end{equation} 
4. Solve the angular flux for the next time step ($\psi^{n+2}$)
	\begin{equation} \begin{split}
	\Omega \cdot \nabla \psi_{u}^{n+2} &+ \left( \Sigma_\mathrm{t} + \frac{1}{v \Delta t} \right) \psi_{u}^{n+2} =  \\ &\Sigma_\mathrm{s} (\phi_{c}^{n+1} + \phi_{u}^{n+1}) + \Sigma_\mathrm{f} (\phi_{c}^{n+1} + \phi_{u}^{n+1}) + Q_{u} + \frac{1}{v \Delta t} \psi_{c}^{n+1} 
	\end{split} \end{equation} 
5. Repeat Steps 1-4 with the new angular flux



<sup>1</sup> K. D. Humbird, J. L. Peterson, and R. G. McClarren. "Deep neural network initialization with decision trees." *IEEE transactions on neural networks and learning systems*,volume 30(5), pp. 1286–1295 (2018)


```python

```
