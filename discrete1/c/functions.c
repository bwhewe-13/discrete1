#include <math.h>
#include "functions.h"

float summation(float arr[],int n){
    float sum = 0;
    for (int ii = 0; ii < n; ii++){
        sum += arr[ii];
    }
    return sum;
}

void normalize(float arr[], float *normal, int n, float maximum, float minimum){
    for (int ii = 0; ii < n; ii++){
        normal[ii] = (arr[ii]-minimum)/(maximum-minimum);
    }
}

void unnormalize(float *arr, float normal[], int n, float maximum, float minimum){
    for (int ii = 0; ii < n; ii++){
        arr[ii] = normal[ii]*(maximum-minimum)+minimum;
    }
}

float find_max(float arr[], int n){
    float max = arr[0];
    for (int ii = 1; ii < n; ii++) {
        if (arr[ii] > max) {
            max = arr[ii];
        }
    }
    return max;
}


float find_min(float arr[], int n){
    float min = arr[0];
    for (int ii = 1; ii < n; ii++) {
        if (arr[ii] < min) {
            min = arr[ii];
        }
    }
    return min;
}

void scale(float *arr, int n, float numerator){
    float denominator = summation(arr,n);
    float scaling = numerator/denominator;
    for (int ii = 0; ii < n; ii++){
        arr[ii] *= scaling;
    }
}

void total_phi(float *mult, float phi[], float xs[], int n){
    for (int ii = 0; ii < n; ii++){
        mult[ii] = phi[ii]*xs[ii];
    }
}

float normalization(float arr1[], float arr2[],int n){
    float num = 0;
    for(int ii = 0; ii < n; ii++){
        num += pow((arr1[ii]-arr2[ii])/arr1[ii],2);
    }
    return sqrt(num);
}

void copy(float *new, float *old, int n){
    for(int ii = 0; ii < n; ii++){
        old[ii] = new[ii];
        // new[ii] = 0;
    }
}