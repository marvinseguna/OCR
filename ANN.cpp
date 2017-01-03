#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include "ANN.h"
#include "Helper.h"
#include <vector>

/*! These 2 arrays are used in the Back-Propagation algorithm in conjunction
with the MOMENTUM constant.
hto = Hidden To Output
ith = Input To Hidden */
double hto[3] = { 0, 0, 0 };
double ith[6] = { 0, 0, 0, 0, 0, 0 };

/*! Initialize ANN weights between each layer
*/
struct ann init_ann() {
	/*! Hidden + Output weights. +1 for the BIAS */
	double* hdn_weights = new double[(INPUT + 1) * HIDDEN];
	double* out_weights = new double[(HIDDEN + 1) * OUTPUT];


	/*! For every hidden node, a connection must exist to each input node + the BIAS! */
	for (int i = 0; i < (INPUT + 1) * HIDDEN; i++) { /*! Storing in 1D array */
		hdn_weights[i] = get_random_value();
	}

	/*! For each output node, a connection mus exist to each hidden node + the BIAS */
	for (int i = 0; i < (HIDDEN + 1) * OUTPUT; i++) {
		out_weights[i] = get_random_value();
	}

	return {hdn_weights, out_weights};
}

/*! Sigmoid function used for the Feed-Forward 
*/
double sigmoid(double value) {
	return 1 / (1 + exp(-value));
}

/*! Forward propagation through the ANN
*/
double* feed_forward(struct ann s_ann, double* inputs) {
	/*! Store each nodes' output in an array as these are required for Back-Propagation */
	double* outputs = new double[(HIDDEN + 1) + OUTPUT];

	/*! Compute outputs from input -> hidden layer. For each hidden node, the computed
	value must then be passed through a sigmoid function
	NOTE: The '+ 1' was added for the BIAS */
	double value = 0;
	for (int i = 0; i < HIDDEN; i++) {
		for (int j = 0; j < INPUT + 1; j++) {
			value += (inputs[j] * s_ann.hidden_layer[(i * (HIDDEN + 1)) + j]);
		}
		outputs[i] = sigmoid(value);
		value = 0;
	}
	outputs[HIDDEN] = 1; /*! Simulating a '1'-value output for BIAS */

	/*! Compute outputs from hidden -> output layer. For each output node, the computed
	value must then be passed through a sigmoid function
	NOTE: The '+ 1' was added for the BIAS */
	for (int i = 0; i < OUTPUT; i++) {
		for (int j = 0; j < HIDDEN + 1; j++) {
			value += (outputs[j] * s_ann.output_layer[(i * (OUTPUT + 1)) + j]);
		}
		/*! 'HIDDEN + 1' since the HIDDEN-index has already been assigned to the BIAS */
		outputs[HIDDEN + 1 + i] = sigmoid(value);
		value = 0;
	}

	return outputs;
}

/*! Computes error for each output node of the ANN
*/
double* calc_output_errors(double expected_output, double* network_outputs, double* delta_outputs) {
	double* output_errors = new double[(HIDDEN + 1) * OUTPUT];

	/*! Formula for output errors:
	1: d(k) = O(k) * (1 - O(k)) * (T(k) - O(k))
	k = current output unit
	d(k) = Delta output for node k
	O(k) = Output of node k
	T(k) = Target output of node k
	2: dw(j,i)(x) = (n * d(j) * x(j,i)) + (m * dw(j,i)(x-1))
	dw(j,i)(x) = current change in weight for connection from node j to node i
	n = Learning rate
	d(j) = Delta output for node in the next layer
	x(j,i) = Input from node i to node j
	m = Momentum
	dw(j,i)(x-1) = previous change in weight for connection from node j to node i */
	for (int i = 0; i < OUTPUT; i++) {
		double node_out = network_outputs[(HIDDEN + 1) + i]; /*! node_out = Current output node value */
		double delta_output = node_out * (1 - node_out) * (expected_output - node_out);
		delta_outputs[i] = delta_output;

		for (int j = 0; j < HIDDEN + 1; j++) {
			int index = (i * (HIDDEN + 1)) + j;
			output_errors[index] = (LEARNING_RATE * delta_output * network_outputs[j]) + (MOMENTUM * hto[index]);

			/*! Value to be used in next iteration with the MOMENTUM */
			hto[index] = output_errors[index];
		}
	}

	return output_errors;
}

/*! Back-Propagation algorithm
*/
struct ann back_propagation(struct ann s_ann, double expected_output, double* network_outputs, double* inputs) {
	double* hidden_errors = new double[(INPUT + 1) * HIDDEN];

	/*! Declare arrays to store the error for each output node. these will be used when calculating
	the error for the hidden nodes */
	double* delta_outputs = new double[OUTPUT];

	/*! Function to calculate the each nodes' error in the output layer */
	double* output_errors = calc_output_errors(expected_output, network_outputs, delta_outputs);

	/*! Formula for hidden errors:
	1: d(h) = O(h) * (1 - O(h)) * sum(all outs)(w(k,h) * delta(k))
	  h = current hidden unit
	  d(h) = Delta output for node h
	  O(h) = Output of node h
	  w(k,h) = weight between node in the next layers' node(k) and this node(h)
	  delta(k) = Delta output for in the next layer(k)
	*/
	for (int i = 0; i < HIDDEN; i++) {
		double value = 0;
		for (int j = 0; j < OUTPUT; j++) {
			value += (s_ann.output_layer[(j * (HIDDEN + 1)) + i] * delta_outputs[j]);
		}
		double delta_hidden = network_outputs[i] * (1 - network_outputs[i]) * value;

		for (int j = 0; j < INPUT + 1; j++) {
			int index = (i * (INPUT + 1)) + j;
			hidden_errors[index] = (LEARNING_RATE * delta_hidden * inputs[j]) + (MOMENTUM * ith[index]);

			/*! Value to be used in next iteration with the MOMENTUM */
			ith[index] = hidden_errors[index];
		}
	}

	/*! Update weights for hidden layer */
	for (int i = 0; i < (HIDDEN + 1) * INPUT; i++) {
		s_ann.hidden_layer[i] += hidden_errors[i];
	}

	/*! Update weights for output layer */
	for (int i = 0; i < (OUTPUT + 1) * HIDDEN; i++) {
		s_ann.output_layer[i] += output_errors[i];
	}

	/*! Free from memory as these are no longer required */
	delete[] output_errors;
	delete[] hidden_errors;
	delete[] delta_outputs;

	return s_ann;
}

/*! Functionality to train the ANN
training_ex: Amount of training examples to iterate on
*/
struct ann train_ann_xor(double* inputs, double* expected_outputs, int training_ex) {
	struct ann s_ann = init_ann();
	double allowance = 0.01;
	double valid_output = 0;

	while (valid_output != training_ex) {
		valid_output = 0;
		for (int i = 0; i < training_ex; i++) {
			int index = i * (INPUT + 1);
			double ann_inputs[3] = { inputs[index], inputs[index + 1], inputs[index + 2] };
			double* outputs = feed_forward(s_ann, ann_inputs);

			if (abs(outputs[3] - expected_outputs[i]) < allowance) {
				valid_output++;
			}
			s_ann = back_propagation(s_ann, expected_outputs[i], outputs, ann_inputs);

			delete[] outputs;
		}
	}

	return s_ann;
}

void learn_xor() {
	/*! Read file from Resources, placing each value in input/output array */
	std::ifstream infile("TestingData.txt");
	double* inputs = (double*)malloc(120 * sizeof(double));
	double* outputs = (double*)malloc(30 * sizeof(double));
	int index = 0;

	while (infile >> inputs[(index * 3)] >> inputs[(index * 3) + 1] >> outputs[index]) {
		inputs[(index * 3) + 2] = 1; /*! BIAS */
		index++;
	}
	realloc(inputs, (index * 3) * sizeof(double));
	realloc(outputs, index * sizeof(double));

	/*! Train ANN*/
	struct ann s_ann = train_ann_xor(inputs, outputs, index);

	delete[] inputs;
	delete[] outputs;
}