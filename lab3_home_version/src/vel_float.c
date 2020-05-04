/* library needed to include the int16_t and int32_t types.
it is needed also in the floating point controllers for the 
definition of outputs and inputs. */
#include <inttypes.h>

#define K 2.6133
#define Ti 0.4523
#define Beta 0.5
#defina h 0.05

/*********************************/
// define controller states here //
/*********************************/
float u = 0.0;
float I = 0.0;
/*
Here use only int16_t variables for the fixed-point 
implementation.
*/

int16_t vel_float(int16_t r, int16_t y){

    /**********************************/
    // Implement your controller here //
    /**********************************/
    
    u = K*β*r - K*y + I;
    I = I + K*h/Ti*(r - y);
    if(u>511)
    {
        u = 511;
    }else if (u < -512)
    {
        u = -512;
    }
    /*
    Use only int32_t and int16_t variables for the fixed-point 
    implementation.
    */

    return (int16_t) u; // write output to D/A and end function
}