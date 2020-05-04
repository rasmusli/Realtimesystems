#include <inttypes.h>

//Values below are calculated in matlab by first using <ss> method to convert given matrices A,B,C to states space
//and the <place>-method to place poles in desired locations and so get K-vector, L-vector, lr, Gamma-vector, and phi-matrix
#define K 2.6133        
#define Ti 0.4523       
#define Beta 0.5        //Set Beta parameter
#define h 0.05          //Set time step parameter h
#define lr 1.7831
#define L1 3.2898
#define L2 1.7831
#define Ke1 2.0361
#define Ke2 1.2440
#define Ke3 3.2096
#define phi_1 0.9940
#define phi_2 0
#define phi_3 0.2493
#define phi_4 1
#define gamma_1 0.1122
#define gamma_2 0.0140

float u = 0.0;          //Control signal
float e = 0.0;          //epsilon, the measured value minus estimated angular position
float x_hat_1 = 0.0;    //Estimation of velocity
float x_hat_2 = 0.0;    //Estimation of angle
float v = 0.0;          //Integral term 

int16_t pos_float(int16_t r, int16_t y){

    u = lr*r - L1*x_hat_1 - L2*x_hat_2 - v; //Calculate u by estimations of vel and angle and integral term
    e = y - x_hat_2;    //new epsilon calculated from measured y and estimated angle
    x_hat_1 = phi_1*x_hat_1 + phi_2*x_hat_2 + gamma_1*(u + v) + Ke1*e; //New estimation of x1
    x_hat_2 = phi_3*x_hat_1 + phi_4*x_hat_2 + gamma_2*(u + v) + Ke2*e; //New estimation of x2
    v = v + Ke3*e;      //New integral term
    
    //Limit control signal to 10 bit D/A
    if (u > 511)
    {
        u = 511;
    }
    else if(u < -512)
    {
        u = -512;
    }

    return (uint16_t) u; 
}