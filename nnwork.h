#ifdef __cplusplus 
extern "C" {
#endif
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

typedef struct {
	unsigned int inum; // number of inputs
	unsigned int hnum; // number of hidden neurons
	unsigned int onum; // number of outputs
	double **ihw; // input to hidden weights
	double **how; // hidden to output weights
	double *hout; // hidden neuron outputs
	double lambda; 
	double rate; 
	double mrate; // frequency of mutations
	double mlow;  // range of impact of mutations
	double mhigh;
} nnwork_t;

// neuron sigmoid
double nnwork_sigmoid(double input, double lambda);

// generate random double between low and high
double randrange(double low, double high);

// initialize new nnwork_t struct
nnwork_t *nnwork_init(unsigned int i, unsigned int h, unsigned int o);

// run a network on inputs
void nnwork_run(nnwork_t*, double*, double*);

// run then adjust a network
void nnwork_train(nnwork_t*, double*, double*);

// breed children of two nnwork_ts
nnwork_t** nnwork_breed(nnwork_t*, nnwork_t*, unsigned short children);

// free a nnwork_t
void nnwork_destroy(nnwork_t *n);

#ifdef __cplusplus 
}
#endif
