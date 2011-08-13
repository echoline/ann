#include <iostream>
#include <fstream>
#include "nnwork.h"
using namespace std;

double get_error(double value, double *outputs) {
	double error = 0.0;

	for (int i = 0; i < 10; i++)
		if (value == i)
			error += (1.0 - outputs[i]) * (1.0 - outputs[i]);
		else
			error += outputs[i] * outputs[i];

	return error;
}

int main(int argc, char **argv) {
	double input[256], output[10];
	unsigned long long epoch = 0;
	nnwork_t *adam;
	nnwork_t *eve;
	nnwork_t **population;

	if (argc < 2) {
		cerr << "please specify training file(s)" << endl;
		return -1;
	}

	srand(time(NULL));

	adam = nnwork_init(256, 100, 10);
 	eve = nnwork_init(256, 100, 10);

	adam->mrate = 0.0001;
	adam->mlow = -4.0;
	adam->mhigh = 4.0;
	eve->mrate = 0.0000000001;
	eve->mlow = -0.5;
	eve->mhigh = 0.5;

	population = nnwork_breed(adam, eve, 20);

	while (true) {
		nnwork_t *first, *second;
		double low = 1E+37, lower = 1E+37;
		double error_levels[20] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
					    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		for (int i = 1; i < argc; i++) {
			ifstream ifs(argv[i]);
			double value;
			short ago = 0;

			while (ifs.good()) {
				ifs >> value;
				if (ago < 256) {
					input[ago] = value;
				} else {
					ago = -1;
					for (int p = 0; p < 20; p++) {
						nnwork_run(population[p], input, output);
						error_levels[p] += get_error(value, output);
					}
				}
				ago++;
			}

			for (int p = 0; p < 20; p++) {
				if (error_levels[p] < lower) {
					low = lower;
					second = first;
					first = population[p];
					lower = error_levels[p];
				} else if (error_levels[p] < low) {
					second = population[p];
					low = error_levels[p];
				}
			}

			epoch++;
			printf("epoch: %lld, first: %lf, second: %lf\n", 
				epoch, lower, low);
			fflush(stdout);

			nnwork_t **new_pop = nnwork_breed(first, second, 20);
			for (int p = 0; p < 20; p++)
				nnwork_destroy(population[p]);
			free(population);
			population = new_pop;
		}
	}
}
