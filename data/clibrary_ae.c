#include <stdio.h>
#include <stdlib.h>
#include <tensorflow/c/c_api.h>
#include <math.h>

#include "functions.h"
#include "autoencode.h"

#define LENGTH 1000 // Number of Spatial Cells
#define GHAT 20 // Number of Energy Levels (Small)
#define G 87 // Number of Energy Levels (Large)


// float* decode_total_encode(float*, float*, float, float);


float* decode_total_encode(float psi[], float xs[], float maxi, float mini){
    // float psi_full[G] = {0};
    float un_psi_full[G] = {0};
    float un_mult[G] = {0};
    float norm_mult[G] = {0};

    // Get value for scaling
    float numerator = summation(psi,GHAT);
    // Decode out to full size
    float* psi_full = decode(psi);
    // Scale the decoder output
    scale(psi_full,G,numerator);
    // Unnormalize the data
    unnormalize(un_psi_full,psi_full,G,maxi,mini);
    // Total * phi
    total_phi(un_mult,un_psi_full,xs,G);
    // Normalize to go back to G_hat
    normalize(un_mult,norm_mult,G,maxi,mini);
    // Get Value for scaling
    numerator = summation(norm_mult,G);
    // Encode down to G_hat
    float* mult = encode(norm_mult);
    // Scale encoder output
    scale(mult,GHAT,numerator);
    return mult;
}


void source_iteration(float *psi_top, float old[], float total_xs[], float psi_bottom[], float Q[], double angle, float maxi, float mini){
    float change = 1;
    int count = 1;
    while(change > 1e-8 && count <= 100){
    // for (int ii = 0; ii < 3; ii++){
        float* alpha_top = decode_total_encode(old,total_xs,maxi,mini);
        float* alpha_bottom = decode_total_encode(psi_bottom,total_xs,maxi,mini);
        // printf("Count %d %f\n",count,change);

        for(int gg = 0; gg < GHAT; gg++){
            psi_top[gg] = (Q[gg] + angle*psi_bottom[gg] + 0.5*alpha_bottom[gg])*old[gg]/(angle*old[gg]-0.5*alpha_top[gg]);
        }
        change = normalization(psi_top,old,GHAT);
        count += 1;
        copy(psi_top,old,GHAT);
    }
    // copy(old,psi_top,GHAT);
}


void sweep_ae(void **flux, void **initialize, void **total, void **outside, void *max_val, void *min_val, double angle){
    float psi_top[GHAT] = {0};
    float psi_bottom[GHAT] = {0};
    
    float ** phi = (float **) flux;
    float ** guess = (float **) initialize;
    float ** cross_section = (float **) total;
    float ** external = (float **) outside;
    float * maxi = (float *) max_val;
    float * mini = (float *) min_val; 

    for(int ii=0; ii < LENGTH; ii++){
        source_iteration(psi_top,guess[ii],cross_section[ii],psi_bottom,external[ii],angle,maxi[ii],mini[ii]);
        // psi_top = (temp_scat[ii] + external[ii] + psi_bottom*top_mult[ii])*bottom_mult[ii];
        for (int jj=0; jj < GHAT; jj++){
            phi[ii][jj] += (angle*0.5*(psi_top[jj]+psi_bottom[jj]));
        }
        copy(psi_top,psi_bottom,GHAT);
        // phi[ii] += (angle*0.5*(psi_top+psi_bottom));
        // psi_bottom = psi_top; 
    }
   
    for(unsigned ii = (LENGTH); ii-- > 0; ){
        // psi_top = psi_bottom;
        copy(psi_bottom,psi_top,GHAT);
        source_iteration(psi_bottom,guess[ii],cross_section[ii],psi_top,external[ii],angle,maxi[ii],mini[ii]);
        // psi_bottom = (temp_scat[ii] + external[ii] + psi_top * top_mult[ii])*bottom_mult[ii];
        // phi[ii] += (angle*0.5*(psi_top+psi_bottom));
        for (int jj=0; jj < GHAT; jj++){
            phi[ii][jj] += (angle*0.5*(psi_top[jj]+psi_bottom[jj]));
        }
    } 
    
}

