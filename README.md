# One-Dimensional Neutron Transport Equation (<img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/1e54c312f0549fca79cfdb38a3588f3d.svg?invert_in_darkmode" align=middle width=21.72608624999999pt height=22.465723500000017pt/>) Solver using the Discrete Ordinates Method

## Machine Learning Methods are used to reduce the data requirements of Scattering and Fission Cross-Sections\ \

### Current Work
- [ ] Add Testing Functions with pytest
- [ ] Clean up source.py and add test functions
- [x] Make time-dependent source problems for source.py
- [ ] Make python to run through benchmark problems (Reed's for Source, LANL Benchmarks for Eigenvalue)

### Current Data Saving Techniques
<!-- 1. <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/6dacb1fb1146dc54364736fb29d4e57a.svg?invert_in_darkmode" align=middle width=86.30068094999999pt height=18.19974420000002pt/> is the correct <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/1e54c312f0549fca79cfdb38a3588f3d.svg?invert_in_darkmode" align=middle width=21.72608624999999pt height=22.465723500000017pt/> code for one-dimensional sweeps. -->
1. dj\_prob.py incorporates Deep Jointly-Informed Neural Networks<sup>1</sup> (DJINN) into the S<sub>N</sub> code for &Sigma;<sub>s</sub> &Phi; and &Sigma;<sub>f</sub> &Phi; calculations.
2. ae\_prob.py incorporates an autoencoder into the S<sub>N</sub> code for &Phi;, &Sigma;<sub>s</sub> &Phi;, and &Sigma;<sub>f</sub> &Phi; to compress the energy groups and use in conjuction with DJINN. 
3. svd\_prob.py incorporates an SVD into the S<sub>N</sub> code for the &Sigma;<sub>s</sub> and &Sigma;<sub>f</sub> matrices.
4. hybrid.py separates the collided and uncollided terms to be used with different numbers of ordinates (N) and energy groups (G). 

### Hybrid Method for Time Dependent Multigroup Problems
0. Initialize <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/a73b9ed978c3a83db5f171415ca083be.svg?invert_in_darkmode" align=middle width=19.42361189999999pt height=22.831056599999986pt/> to zero
1. Calculate the uncollided <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/c77f3b1de554cb37e2114a29ce9cae87.svg?invert_in_darkmode" align=middle width=36.06753149999999pt height=26.76175259999998pt/> and <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/71575e9dddc7fe2a251de8f75f80a141.svg?invert_in_darkmode" align=middle width=34.56448709999999pt height=26.76175259999998pt/> through the sweep
	<p align="center"><img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/defec23a4277b3ee2e5d3c1d7b7059f6.svg?invert_in_darkmode" align=middle width=522.28458315pt height=39.452455349999994pt/></p> <p align="center"><img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/ea93f4ca745a4c5420d38964dc1585e7.svg?invert_in_darkmode" align=middle width=561.5863671pt height=85.48022999999999pt/></p> 
2. Use <img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/71575e9dddc7fe2a251de8f75f80a141.svg?invert_in_darkmode" align=middle width=34.56448709999999pt height=26.76175259999998pt/> to create source term (<img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/b827948b03567379a24e18fa5edaedd1.svg?invert_in_darkmode" align=middle width=18.870076499999993pt height=22.465723500000017pt/>) for the collided equation: 
	<p align="center"><img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/486e18dcba6b246d55f231fcc75e1008.svg?invert_in_darkmode" align=middle width=434.16891375pt height=18.312383099999998pt/></p>
3. Solve the collided equation with the new source term (<img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/b827948b03567379a24e18fa5edaedd1.svg?invert_in_darkmode" align=middle width=18.870076499999993pt height=22.465723500000017pt/>)
	<p align="center"><img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/fd31c140f3073c02d234ac565f5f88b3.svg?invert_in_darkmode" align=middle width=558.37131075pt height=39.452455349999994pt/></p> 
4. Solve the angular flux for the next time step (<img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/db4105c0603e7d298bc88eb06b421593.svg?invert_in_darkmode" align=middle width=36.06753149999999pt height=26.76175259999998pt/>)
	<p align="center"><img src="https://rawgit.com/bwhewe-13/discrete1 (fetch/master/svgs/f49edd9196a68e14e720c72c80650161.svg?invert_in_darkmode" align=middle width=584.19276795pt height=79.0179555pt/></p> 
5. Repeat Steps 1-4 with the new angular flux



<sup>1</sup>:  K. D. Humbird, J. L. Peterson, and R. G. McClarren. "Deep neural network initialization with decision trees." *IEEE transactions on neural networks and learning systems*,volume 30(5), pp. 1286–1295 (2018)
