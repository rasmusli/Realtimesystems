
#include <inttypes.h>

#define n 12                //Number of fractional bits, fixed point arithmetics made in << 12, see vel_fixed.c to see why 12 was used
//Values below are calculated in matlab by first using <ss> method to convert given matrices A,B,C to states space
//and the <place>-method to place poles in desired locations and so get K-vector, L-vector, lr, Gamma-vector, and phi-matrix
//All of the below values are in fp, meaning they are shifted up n bits
#define Ke1_fp 8340         
#define Ke2_fp 5095         
#define Ke3_fp 13135        
#define lr_fp 7304          
#define L1_fp 13475         
#define L2_fp 7304          
#define gam_1_fp 460
#define gam_2_fp 57
#define phi_1_fp 4071
#define phi_2_fp 0
#define phi_3_fp 1021
#define phi_4_fp 4096



int16_t u = 0;          //Control signal
int16_t e = 0;          //epsilon, the measured value minus estimated angular position
int16_t x_h_1 = 0;      //Estimation of velocity
int16_t x_h_2 = 0;      //Estimation of angle
int16_t v = 0;          //Integral term 

int16_t pos_fixed(int16_t r, int16_t y){

    //all the terms used to calculate u
    int32_t u_term1 = ((int32_t) lr_fp) *  r; //fixed point lr times reference value
    int32_t u_term2 = ((int32_t)L1_fp) * x_h_1; //fixed point L1 times estimated vel
    int32_t u_term3 = ((int32_t) L2_fp) * x_h_2; //fixed point L2 times estimated angle


    u = (u_term1 - u_term2 - u_term3 - v ) >> n; //Calculating u which not in fixed point

    e = y - x_h_2; //Calculating epsilon, which is the measured value minus estimated angular position

    //All terms used to calculate x_h_1, which is the estimation of velocity
    int32_t x1_term1 = ((int32_t)phi_1_fp) * x_h_1; //fixed point phi11 times estimated vel
    int32_t x1_term2 = ((int32_t)phi_2_fp) * x_h_2; //fixed point phi12 times estimated angle
    int32_t x1_term3 = ((int32_t)gam_1_fp) * (u + v) ; //fixed point gamma1 times control signal and integral term
    int32_t x1_term4 = ((int32_t)Ke1_fp) * e;         //fixed point first element of K-vector times epsilon

    //All terms used for calculating new x_h_2, which is the estimation of angle
    int32_t x2_term1 = ((int32_t)phi_3_fp) * x_h_1; //fixed point phi21 times estimated vel
    int32_t x2_term2 = ((int32_t)phi_4_fp) * x_h_2 ; //fixed point phi22 times estimated angle
    int32_t x2_term3 = ((int32_t)gam_2_fp) * (u + v); //fixed point gamma2 times control signal and integral term
    int32_t x2_term4 = ((int32_t)Ke2_fp) * e; //fixed point second element of K-vector times epsilon

    x_h_1 = (x1_term1 + x1_term2 + x1_term3 + x1_term4) >> n; //New estimation of x_1 for t+1
    x_h_2 = (x2_term1 + x2_term2 + x2_term3 + x2_term4) >> n; //New estimation of x_2 for t+1

    int32_t K_ve = (((int32_t)Ke3_fp)*e) >> n; //Third element of K-vector times epsilon, then shifting down by n
    v = v + K_ve; //New integral term for t+1

    //Limit control signal to 10 bit D/A
    if (u > 511)
    {
        u = 511;
    }
    else if(u < -512)
    {
        u = -512;
    }
    

	return u;
}