#include <stdio.h>
#include <stdlib.h>
// #define LENGTH 1000

#ifndef HDPE
#define HDPE 500
#endif

#ifndef PU239
#define PU239 150
#endif

#ifndef PU240 
#define PU240 350
#endif


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


void reflected_reduced(void *flux, void *guess, void *scatter, void *external, void *numerator, void *denominator, double weight){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * phi_old = (double *) guess;
    double * temp_scat = (double *) scatter;       // of length materials

    double * source = (double *) external;         // of length spatial 
    double * top_mult = (double *) numerator;      // of length materials
    double * bottom_mult = (double *) denominator; // of length materials

    // Has to change with each orientation
    int materials = 3;
    int widths[3] = {HDPE,PU239,PU240};
    int global_index = 0;

    for(int mat=0; mat < materials; mat++){
        for(int ii=0; ii < widths[mat]; ii++){
            psi_top = (temp_scat[mat] * phi_old[global_index] + source[global_index] + psi_bottom * top_mult[mat]) * bottom_mult[mat];
            // Write flux to variable
            phi[global_index] += (weight *0.5 *(psi_top + psi_bottom));
            // Move to next cell
            psi_bottom = psi_top; 
            // Update global index
            global_index += 1;
        }
    }

    for (unsigned mat = materials; mat-- > 0; ){
        for(unsigned ii = widths[mat]; ii-- > 0; ){ // Sweep back from right to left
            // Update global index
            global_index -= 1;
            // Start at i = I
            psi_top = psi_bottom;
            psi_bottom = (temp_scat[mat] * phi_old[global_index] + source[global_index] + psi_top * top_mult[mat]) * bottom_mult[mat];
            //  Write flux to variable
            phi[global_index] += (weight * 0.5 * (psi_top + psi_bottom));
        }
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
