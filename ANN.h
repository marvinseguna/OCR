#pragma once

// Begin - CLASSES
/*! ANN structure 
*/
struct ann {
	double* hidden_layer;
	double* output_layer;
};
// End - CLASSES

//------------------------------------------//

// Begin - CONSTANTS
#define INPUT 2 /*! Amount of inputs nodes(2) */
#define HIDDEN 2 /*! Amount of hidden nodes (2) */
#define OUTPUT 1 /*! Amount of output nodes (1) */
#define LEARNING_RATE 0.5 /*! Constant used for Back-Propagation */
#define MOMENTUM 0.8 /*! Constant used for Back-Propagation */
// End - CONSTANTS

//------------------------------------------//

// Begin - METHODS
void learn_xor();
// End - METHODS
