#include <omp.h>
#include <iostream>
#include <fstream>
#include <map>
#include "nnwork.h"
using namespace std;

nnwork_t* select_one(map<double, nnwork_t*> *error_map) {
	int p;

	while (true) {
		for (map<double, nnwork_t*>::iterator it = error_map->begin(); it != error_map->end(); it++) {
			p = (int)randrange(0, 1.618);
			if (p == 0) {
				error_map->erase(it);
				return it->second;
			}
		}
	}
}

double get_error(double value, double *outputs) {
	double error = 0.0;

	for (int i = 0; i < 1; i++)
		if (value == i)
			error += (1.0 - outputs[i]) * (1.0 - outputs[i]);
		else
			error += outputs[i] * outputs[i];

	return error;
}

int main(int argc, char **argv) {
	double input[2], output[1];
	unsigned long long epoch = 0;
	nnwork_t *first;
	nnwork_t *second;
	nnwork_t **population;

	if (argc < 2) {
		cerr << "please specify training file(s)" << endl;
		return -1;
	}

	srand(time(NULL));

	first = nnwork_init(2, 10, 1);
 	second = nnwork_init(2, 10, 1);
	population = nnwork_breed(first, second, 50);
	nnwork_destroy(first);
	nnwork_destroy(second);

	while (true) {
		map<double, nnwork_t*> error_map;
		double error_levels[20] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
					    0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		for (int i = 1; i < argc; i++) {
			ifstream ifs(argv[i]);
			double value;
			short ago = 0;

			while (ifs.good()) {
				ifs >> value;
				if (ago < 2) {
					input[ago] = value;
				} else {
					ago = -1;
					#pragma omp parallel for private(output)
					for (int p = 0; p < 20; p++) {
						nnwork_run(population[p], input, output);
						error_levels[p] += get_error(value, output);
					}
				}
				ago++;
			}

			for (int p = 0; p < 20; p++)
				error_map.insert(pair<double,nnwork_t*>(error_levels[p], population[p]));

			printf("epoch: %lld", epoch++);
			for(map<double, nnwork_t*>::iterator it = error_map.begin(); it != error_map.end(); it++)
				printf(", %lf", it->first);
			printf("\n");
			fflush(stdout);

			first = error_map.begin()->second;
			error_map.erase(error_map.begin());
			second = select_one(&error_map);

			nnwork_t **new_pop = nnwork_breed(first, second, 50);
			for (int p = 0; p < 20; p++)
				nnwork_destroy(population[p]);
			free(population);
			population = new_pop;
		}
	}
}
