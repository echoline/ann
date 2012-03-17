#include <iostream>
#include <fstream>
#include "nnwork.h"
using namespace std;

void set_outputs(double value, double *output) {
	for (int i = 0; i < 1; i++)
		if (value == i)
			output[i] = 1.0;
		else
			output[i] = 0.0;
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
	nnwork_t *ann = nnwork_init(2, 10, 1);
	double input[2], output[1];
	unsigned long long epoch = 0;

	if (argc < 2) {
		cerr << "please specify training file(s)" << endl;
		return -1;
	}

	srand(time(NULL));

	while (true) {
		for (int i = 1; i < argc; i++) {
			ifstream ifs(argv[i]);
			double value;
			long ago = 0;

			while (ifs.good()) {
				ifs >> value;
				if (ago < 2) {
					input[ago] = value;
				} else {
					ago = -1;
					//set_outputs(value, output);
					nnwork_train(ann, input, &value); //output);
					epoch++;
				}
				ago++;
			}
			nnwork_run(ann, input, output);
			printf("epoch: %lld, error level: %lf\n", epoch, get_error(value, output));
			fflush(stdout);

		}
	}
}
