#include <stdio.h>
#include <stdlib.h>

#ifndef GROUPS
#define GROUPS 87
#endif

#ifndef CELLS
#define CELLS 1000
#endif

#ifndef ANGLES
#define ANGLES 8
#endif

#ifndef MATERIALS
#define MATERIALS 1
#endif


// For total cross sections
typedef struct xs_vector {
    double array[MATERIALS][GROUPS];
} xs_vector;
// For scatter and fission cross sections
typedef struct xs_matrix {
    double array[MATERIALS][GROUPS][GROUPS];
} xs_matrix;
// For boundary conditions
typedef struct boundary_edges {
    double array[GROUPS][2];
} boundary_conditions;
// For source materials
typedef struct spatial_energy {
    double array[GROUPS][CELLS];
} spatial_energy;
// For cell width (eventually) and material map
// typedef struct spatial

/*
void slab(void *flux, void *psi_angle, void *scatter, void *external, void *numerator, 
                void *denominator, double weight, int direction){

    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * psi = (double *) psi_angle; 
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    for(int ii=0; ii < CELLS; ii++){ // Sweep from left to right
      psi_top = (temp_scat[ii] + source[ii] + psi_bottom * top_mult[ii]) * bottom_mult[ii];
      // Write flux to variable
      phi[ii] += (weight *0.5 *(psi_top + psi_bottom));
      // Move to next cell
      psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (CELLS); ii-- > 0; ){ // Sweep back from right to left
      // Start at i = I
      psi_top = psi_bottom;
      psi_bottom = (temp_scat[ii] + source[ii] + psi_top * top_mult[ii]) * bottom_mult[ii];
      //  Write flux to variable
      phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
    } 
    
}
*/


void reflected(void *flux, void *psi_angle, void *scatter, void *external, void *numerator, 
                void *denominator, double weight, int direction){

    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * psi = (double *) psi_angle; 
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    for(int ii=0; ii < CELLS; ii++){ // Sweep from left to right
      psi_top = (temp_scat[ii] + source[ii] + psi_bottom * top_mult[ii]) * bottom_mult[ii];
      // Write flux to variable
      phi[ii] += (weight *0.5 *(psi_top + psi_bottom));
      // Move to next cell
      psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (CELLS); ii-- > 0; ){ // Sweep back from right to left
      // Start at i = I
      psi_top = psi_bottom;
      psi_bottom = (temp_scat[ii] + source[ii] + psi_top * top_mult[ii]) * bottom_mult[ii];
      //  Write flux to variable
      phi[ii] += (weight * 0.5 * (psi_top + psi_bottom));
    } 
    
}


void vacuum(void *flux, void *psi_angle, void *scatter, void *external, void *numerator,
             void *denominator, double weight, int direction){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * psi = (double *) psi_angle; 
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    if(direction == 1){
        psi_bottom = 0.0; // source enters from LHS
        for(int ii=0; ii < CELLS; ii++){
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
        for(unsigned ii = (CELLS); ii-- > 0; ){
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