# One-Dimensional Neutron Transport Equation (S<sub>N</sub>)  Solver using the Discrete Ordinates Method

## Machine Learning Methods are used to reduce the data requirements of Scattering and Fission Cross-Sections

### Current Work
- [ ] Add Testing Functions with pytest
- [ ] Clean up criticality (k-eigenvalue problems)
- [ ] Make time-dependent source problems for source.py
- [ ] Update the Benchmark File
- [ ] Add 1D spherical problems for both time-dependent source problems and k-eigenvalue problems
- [ ] Update and remove functions and files no longer needed
- [ ] Create doc files for LaTeX files on the math

### Current Data Saving Techniques
1. dj\_prob.py incorporates Deep Jointly-Informed Neural Networks<sup>1</sup> (DJINN) into the S<sub>N</sub> code for &Sigma;<sub>s</sub> &Phi; and &Sigma;<sub>f</sub> &Phi; calculations.
2. ae\_prob.py incorporates an autoencoder into the S<sub>N</sub> code for &Phi;, &Sigma;<sub>s</sub> &Phi;, and &Sigma;<sub>f</sub> &Phi; to compress the energy groups and use in conjuction with DJINN. 
3. svd\_prob.py incorporates an SVD into the S<sub>N</sub> code for the &Sigma;<sub>s</sub> and &Sigma;<sub>f</sub> matrices.
4. hybrid.py separates the collided and uncollided terms to be used with different numbers of ordinates (N) and energy groups (G). 

### Python Files Being Updated
- [ ] dj\_prob.py, svd\_prob.py, and ae\_prob.py &#8594; reduction.py
- [ ] setup\_ke.py &#8594; keigenvalue.py
- [ ] correct.py &#8594; critical.py

### Python Files to Remove
Important functions have to be extracted before these are removed
- [ ] util.py
- [ ] KEproblems.py

<sup>1</sup> K. D. Humbird, J. L. Peterson, and R. G. McClarren. "Deep neural network initialization with decision trees." *IEEE transactions on neural networks and learning systems*,volume 30(5), pp. 1286–1295 (2018)
