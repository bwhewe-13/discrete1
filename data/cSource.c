#include <stdio.h>
#include <stdlib.h>
#define LENGTH 1000
// #define LENGTH 1

void reflected(void *flux, void *scatter, void *external, void *numerator, void *denominator, double weight, int direction){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    for(int ii=0; ii < LENGTH; ii++){ // Sweep from left to right
      psi_top = (temp_scat[ii] + source[ii] + psi_bottom * top_mult[ii]) * bottom_mult[ii];
      // Write flux to variable
      phi[ii] += (weight *0.5 *(psi_top + psi_bottom));
      // Move to next cell
      psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (LENGTH); ii-- > 0; ){ // Sweep back from right to left
      // Start at i = I
      psi_top = psi_bottom;
      psi_bottom = (temp_scat[ii] + source[ii] + psi_top * top_mult[ii]) * bottom_mult[ii];
      //  Write flux to variable
      phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
    } 
    
}



void vacuum(void *flux, void *scatter, void *external, void *numerator, void *denominator, double weight, int direction){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    if(direction == 1){
        for(int ii=0; ii < LENGTH; ii++){
           psi_top = (temp_scat[ii] + source[ii] + psi_bottom * top_mult[ii]) * bottom_mult[ii];
           // Write flux to variable
           phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
           // Move to next cell
           psi_bottom = psi_top; 
        }
    }

    else if (direction == -1){
        for(unsigned ii = (LENGTH); ii-- > 0; ){
            // Start at i = I
            psi_top = psi_bottom;
            psi_bottom = (temp_scat[ii] + source[ii] + psi_top * top_mult[ii]) * bottom_mult[ii];
            // Write flux to variable
            phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
        }
    }

}

// Time Dependent One Group (Vacuum)
void time_vacuum(void *flux, void *psi_angle, void *scatter, void *external, void *numerator, void *denominator, double weight, int direction){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * psi = (double *) psi_angle; 
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    if(direction == 1){
        for(int ii=0; ii < LENGTH; ii++){
           psi_top = (temp_scat[ii] + source[ii] + psi_bottom * top_mult[ii]) * bottom_mult[ii];
           // Write psi to variable
           psi[ii] = 0.5* (psi_top + psi_bottom);
           // Write flux to variable
           phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
           // Move to next cell
           psi_bottom = psi_top; 
        }
    }

    else if (direction == -1){
        for(unsigned ii = (LENGTH); ii-- > 0; ){
            // Start at i = I
            psi_top = psi_bottom;
            psi_bottom = (temp_scat[ii] + source[ii] + psi_top * top_mult[ii]) * bottom_mult[ii];
            // Write psi to variable
            psi[ii] = 0.5 * (psi_top + psi_bottom);
            // Write flux to variable
            phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
        }
    }

}