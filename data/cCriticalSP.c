#include <stdio.h>
#include <stdlib.h>
// #define LENGTH 1000
// #define N 64



void vacuum(void *flux, void *half, void *external, void *v_total, void *area_one, void *area_two, void *dir, void *wgt){
  // Scalar and angular flux
  double * phi = (double *) flux;
  double * psi_nhalf = (double *) half;
  // Scattering and Total Terms
  double * source = (double *) external;
  double * total = (double *) v_total;
  // Surface Area
  double * SA_plus = (double *) area_one;
  double * SA_minus = (double *) area_two; 
  // Angles
  double * mu = (double *) dir;
  double * w = (double *) wgt; 
  
  // double * center = (double *) middle;

  double psi, psi_ihalf;
  double alpha_plus;

  double center[N] = {0};
  double alpha_minus = 0.; // For first angle

  for(int n=0; n < N; n++){

    if (n == N - 1){
      alpha_plus = 0.;
    } 
    else {
      alpha_plus = alpha_minus - (mu[n] * w[n]);
    }

    // Sweep from edge to center
    if (mu[n] < 0){
          
      psi_ihalf = 0; // Vacuum at edge

      for(unsigned ii = (LENGTH); ii-- > 0; ){
        psi = (-1 * mu[n] * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w[n] * (SA_plus[ii] - SA_minus[ii]) 
              * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
              (-2 * mu[n] * SA_minus[ii] + 2/w[n] * (SA_plus[ii] - SA_minus[ii]) * 
                  (alpha_plus) + total[ii]);
        
        if(ii == 0){
          center[n] = psi;
        }
        // Update flux
        phi[ii] = phi[ii] + (w[n] * psi);
        // Update spatial and angle bounds
        psi_ihalf = 2 * psi - psi_ihalf;
        psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];

      }
    }

    else if (mu[n] > 0){

      psi_ihalf = center[N-n-1]; // Reflected at center
      // psi_ihalf = 0;

      for(int ii=0; ii < LENGTH; ii++){
        psi = (mu[n] * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w[n] * (SA_plus[ii] - SA_minus[ii]) 
               * (alpha_plus + alpha_minus) * psi_nhalf[ii] +  source[ii]) / 
              (2 * mu[n] * SA_plus[ii] + 2/w[n] * (SA_plus[ii] - SA_minus[ii]) * 
                  (alpha_plus) + total[ii]);
        // Update spatial cell at middle
        if(ii == 0){
          center[n] = psi;
        }
        // Update flux
        phi[ii] = phi[ii] + (w[n] * psi);
        // Update spatial and angle bounds
        psi_ihalf = 2 * psi - psi_ihalf;
        psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];

      }
    }
  alpha_minus = alpha_plus;

  }

}

void one_direction(void *flux, void *half, void *external, void *v_total, void *area_one, void *area_two, double w, double mu, double alpha_plus, double alpha_minus, int sweep){
  // Scalar and angular flux
  double * phi = (double *) flux;
  double * psi_nhalf = (double *) half;
  // Scattering and Total Terms
  double * source = (double *) external;
  double * total = (double *) v_total;
  // Surface Area
  double * SA_plus = (double *) area_one;
  double * SA_minus = (double *) area_two; 
  
  double psi_ihalf = 0;
  double psi = 0;

  // Sweep from edge to center
  if(sweep == 1){

    psi_ihalf = 0; // Vacuum at edge

    for(int ii=0; ii < LENGTH; ii++){
      psi = (mu * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w * (SA_plus[ii] - SA_minus[ii]) 
            * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
            (2 * mu * SA_plus[ii] + 2/w * (SA_plus[ii] - SA_minus[ii]) * 
                (alpha_plus) + total[ii]);

      
      // Update flux
      phi[ii] += (w * psi);
      // phi[ii] = 1;
      // Update spatial and angle bounds
      psi_ihalf = 2 * psi - psi_ihalf;
      psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];
    }
  }

  else if (sweep == -1){

    psi_ihalf = 0; // Vacuum at edge

    for(unsigned ii = (LENGTH); ii-- > 0; ){
      psi = (-1 * mu * (SA_plus[ii] + SA_minus[ii]) * psi_ihalf + 1/w * (SA_plus[ii] - SA_minus[ii]) 
            * (alpha_plus + alpha_minus) * psi_nhalf[ii] + source[ii]) /
            (-2 * mu * SA_minus[ii] + 2/w * (SA_plus[ii] - SA_minus[ii]) * 
                (alpha_plus) + total[ii]);
      // Update flux
      phi[ii] += (w * psi);
      // Update spatial and angle bounds
      psi_ihalf = 2 * psi - psi_ihalf;
      psi_nhalf[ii] = 2 * psi - psi_nhalf[ii];
    }
  }

}
