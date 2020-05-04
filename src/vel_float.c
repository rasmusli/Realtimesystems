/* library needed to include the int16_t and int32_t types.
it is needed also in the floating point controllers for the 
definition of outputs and inputs. */
#include <inttypes.h>

//Values below are calculated in matlab by using MATLAB equation solver for denominator = 0 for closed loop system given by control processes C and P
#define K 2.6133        
#define Ti 0.4523      
#define Beta 0.5     
#define h 0.05          


float u = 0.0;  //Control signal
float I = 0.0;  //integral term

int16_t vel_float(int16_t r, int16_t y)
{   
    
    
    u = K*Beta*r - K*y + I;             // Calculates new control signal
    I = I + K*h/Ti*(r-y);               // New integral term for t+1
    
    //Limit control signal to 10 bit D/A
    if (u > 511)                        
    {
        u = 511;                        
    }
    else if (u < -512)                  
    {
        u = -512;                       
    }

    return (int16_t) u; 
}