/* library needed to include the int16_t and int32_t types.
it is needed also in the floating point controllers for the 
definition of outputs and inputs. */
#include <inttypes.h>

#define K 9.99  // examle of how parameters are defined in C

/*********************************/
// define controller states here //
/*********************************/
/*
Here use only int16_t variables for the fixed-point 
implementation.
*/

int16_t pos_float(int16_t r, int16_t y){

    /**********************************/
    // Implement your controller here //
    /**********************************/
    /*
    Use only int32_t and int16_t variables for the fixed-point 
    implementation.
    */

    return 255; // write output to D/A and end function
}