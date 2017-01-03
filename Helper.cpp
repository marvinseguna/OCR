#include <stdlib.h>
#include <math.h>
#include "ANN.h"
#include "Helper.h"

double get_random_value() {
	double random = rand() % 100 + 1;
	return (random / 100) - 0.5;
}