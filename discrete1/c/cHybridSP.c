#include <stdio.h>
#include <stdlib.h>
// #define LENGTH 1000
#define PI 3.141592653589793

void surface_area_calc(double surface_area[], double delta){
  // double surface_area[LENGTH + 1];
  for(int ii=0; ii < (LENGTH + 1); ii++){
    surface_area[ii] = 4 * PI * (ii * delta) * (ii * delta);
  }
  
}

// Time Independent Problems
void sweep(void *psif, void *flux, void *half, void *external, void *v_total, 
           double delta, double w, double mu, double alpha_plus, 
           double alpha_minus, double psi_ihalf, double tau){
  // Scalar and angular flux
  double * angular = (double *) psif;
  double * phi = (double *) flux;
  double * psi_nhalf = (double *) half;
  // Scattering and Total Terms
  double * source = (double *) external;
  double * total = (double *) v_total;
  // Surface Area
  // double * SA_plus = (double *) area_one;
  // double * SA_minus = (double *) area_two; 

  double SA[LENGTH + 1];
  surface_area_calc(SA, delta);
  
  double psi;
  
  if(mu > 0){ // Center to edge

    for(int ii=0; ii < LENGTH; ii++){
      // psi = (mu * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w * (SA_plus[ii] - SA_minus[ii]) 
      //       * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
      //       (2 * mu * SA_plus[ii] + 2/w * (SA_plus[ii] - SA_minus[ii]) * 
      //           (alpha_plus) + total[ii]);

      psi = (mu * (SA[ii+1] + SA[ii]) * psi_ihalf + 1/w * (SA[ii+1] - SA[ii]) 
      * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
      (2 * mu * SA[ii+1] + 2/w * (SA[ii+1] - SA[ii]) * 
          (alpha_plus) + total[ii]);

      // Collecting Angular Flux
      angular[ii] = psi;
      // Update flux
      phi[ii] += (w * psi);
      // Update spatial and angle bounds
      psi_ihalf = 2 * psi - psi_ihalf;
      // psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];
      if (ii != 0){
        // Lewis and Miller Corrector
        // psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];

        // Morel and Montry Corrector
        psi_nhalf[ii] = 1/tau * (psi - (1 - tau) * psi_nhalf[ii]);
      }
    }
  }

  else if (mu < 0){ // Edge to center

    for(unsigned ii = (LENGTH); ii-- > 0; ){
      // psi = (-1 * mu * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w * (SA_plus[ii] - SA_minus[ii]) 
      //       * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
      //       (-2 * mu * SA_minus[ii] + 2/w * (SA_plus[ii] - SA_minus[ii]) * 
      //           (alpha_plus) + total[ii]);

      psi = (-1 * mu * (SA[ii+1] + SA[ii]) * psi_ihalf + 1/w * (SA[ii+1] - SA[ii]) 
            * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
            (-2 * mu * SA[ii] + 2/w * (SA[ii+1] - SA[ii]) * 
                (alpha_plus) + total[ii]);
            
      // Collecting Angular Flux
      angular[ii] = psi;
      // Update flux
      phi[ii] += (w * psi);
      // Update spatial and angle bounds
      psi_ihalf = 2 * psi - psi_ihalf;
      // psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];
      if (ii != 0){
        // Lewis and Miller Corrector
        // psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];

        // Morel and Montry Corrector
        psi_nhalf[ii] = 1/tau * (psi - (1 - tau) * psi_nhalf[ii]);
      }
    }
  }

}
