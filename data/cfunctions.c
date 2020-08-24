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