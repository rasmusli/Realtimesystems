/* library needed to include the int16_t and int32_t types.
it is needed also in the floating point controllers for the 
definition of outputs and inputs. */
#include <inttypes.h>

#define K 2.6133        //
#define Ti 0.4523       //Defines Ti parameter
#define Beta 0.5        //Defines Beta parameter
#define h 0.05          //Defines time step parameter h
#define n = 5
#define uint32_t KBeta 42

/*********************************/
// define controller states here //
/*********************************/
/*
Here use only int16_t variables for the fixed-point 
implementation.
*/
static int32_t u = 0;
static int32_t I = 0;
static int32_t K_fp = 84;
static int32_t Beta_fp = 16;
static int32_t KBeta_fp = 42;
static int32_t Kh_fp = 4;


int16_t vel_fixed(int16_t r, int16_t y){

    /**********************************/
    // Implement your controller here //
    /**********************************/
    /*
    Use only int32_t and int16_t variables for the fixed-point 
    implementation.
    */
    
    u = KBeta_fp*r - K_fp*y + I;             // Calculates the control signal u
    I = I + Kh_fp/Ti*(r-y);                  // Calculates the Integral term for t+1

	return u >> n; // write output to D/A and end function
}