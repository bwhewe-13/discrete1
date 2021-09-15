#include <stdio.h>
#include <stdlib.h>
#include "functions.h"

#ifndef I
#define I 1000
#endif

#ifndef N
#define N 8
#endif

#ifndef G
#define G 87
#endif

#ifndef materials 
#define materials 1
#endif

// Define 2D and 3D Array sizes
typedef struct psi_matrix {
    double array[I][N];
} psi_matrix;

typedef struct boundary {
    double array[G][2]
} boundary;

typedef struct multi_vec {
    double array[materials][G];
} multi_vec;

typedef struct multi_mat {
    double array[materials][G][G];
} multi_mat;


extern void multigroup()

void sweep(void *flux, void *scatter, void *external, void *numerator, void *denominator, double weight, double boundary, int direction){
    double psi_top;
    double psi_bottom = boundary;
    
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



void vacuum(void *flux, void *scatter, void *external, void *numerator, void *denominator, double weight, double boundary, int direction){
    double psi_top;
    double psi_bottom = 0.0;

    double * phi = (double *) flux;
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    if(direction == 1){
        psi_bottom = boundary; // source enters from LHS
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
void time_vacuum(void *flux, void *psi_angle, void *scatter, void *external, void *numerator, void *denominator, double weight, double boundary, int direction){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * psi = (double *) psi_angle; 
    double * temp_scat = (double *) scatter;

    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    if(direction == 1){
        psi_bottom = boundary; // source enters from LHS
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