/* library needed to include the int16_t and int32_t types.
it is needed also in the floating point controllers for the 
definition of outputs and inputs. */
#include <inttypes.h>

#define n 12             //Shift value for fixed point calculations, I used a large number to get a higher accuracy on the calculations
                        // It could probably have been a bit lower and still be good enough, 
                        // but since it's not such a heavy calculation I decided to be on the safe side
//Values below are calculated in matlab by using MATLAB equation solver for denominator = 0 for closed loop system given by control processes C and P
//All values below are in 
#define K_fp 10704         
#define Beta_fp 2048      
#define KBeta_fp 5352     //Combined value of K and Beta in fp, in order to save calculation time and increase accuracy
#define Kh_fp 535         //Combined value of K and h, in order to save calculation time and increase accuracy
#define Ti_fp 1792        

static int16_t u = 0;   //Control signal
static int16_t I = 0;   //integral term


int16_t vel_fixed(int16_t r, int16_t y){

    //The terms below are used for calculating control signal u
    int32_t u_term1 = (KBeta_fp*r) >> n;        
    int32_t u_term2 = (K_fp*y) >> n;             
    
    u = u_term1 - u_term2 + I;                        // Control signal u

    
    int32_t I_c = (Kh_fp << n)/Ti_fp;   // I constant term. Shifting up Kh first and and doing the division, to increase accuracy (fp*fp/fp = fp)
    int32_t I_t1 = (I_c*(r-y)) >> n;   // New integral term to add to I t+1

    I = I + I_t1;                          // Calculates the Integral term for I t+1

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