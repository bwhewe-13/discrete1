#include <stdio.h>
#include <stdlib.h>
#define LENGTH 1000

void sweep(void *flux, void *scatter, void *outside, void *total1, void *total2, double angle){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * temp_scat = (double *) scatter;
    double * external = (double *) outside;
    double * top_mult = (double *) total1;
    double * bottom_mult = (double *) total2; 

    for(int ii=0; ii < LENGTH; ii++){
       psi_top = (temp_scat[ii] + external[ii] + psi_bottom*top_mult[ii])*bottom_mult[ii];
       phi[ii] += (angle*0.5*(psi_top+psi_bottom));
       psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (LENGTH); ii-- > 0; ){
        psi_top = psi_bottom;
        psi_bottom = (temp_scat[ii] + external[ii] + psi_top * top_mult[ii])*bottom_mult[ii];
        phi[ii] += (angle*0.5*(psi_top+psi_bottom));
    } 
    
}



// Uncollided One Group
void uncollided(void *flux, void *external, void *numerator, void *denominator, void *front, void *back, double angle){
    double psi_top;
    double psi_bottom = 0.0;
    
    double * phi = (double *) flux;
    double * source = (double *) external;
    double * top_mult = (double *) numerator;
    double * bottom_mult = (double *) denominator; 

    double * psi_last_front = (double *) front;
    double * psi_last_back = (double *) back;

    for(int ii=0; ii < LENGTH; ii++){
       psi_top = (source[ii] + psi_bottom*top_mult[ii])*bottom_mult[ii];
       psi_last_front[ii] = 0.5*(psi_top+psi_bottom);
       phi[ii] += (angle*0.5*(psi_top+psi_bottom));
       psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (LENGTH); ii-- > 0; ){
        psi_top = psi_bottom;
        psi_bottom = (source[ii] + psi_top * top_mult[ii])*bottom_mult[ii];
        psi_last_back[ii] = 0.5*(psi_top+psi_bottom);
        phi[ii] += (angle*0.5*(psi_top+psi_bottom));
    } 
    
}

