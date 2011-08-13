#include "nnwork.h"

// neuron sigmoid
double nnwork_sigmoid(double input, double lambda) {
	return (1.0 / (1.0 + exp(-input*lambda)));
}

// generate random double between low and high
double randrange(double low, double high) {
	return (high - low) * (rand() / (double)(RAND_MAX)) + low;
}

// initialize new nnwork_t struct
nnwork_t *nnwork_init(unsigned int inum, unsigned int hnum, unsigned int onum) {
	unsigned int i, h, o;
	nnwork_t *ret = (nnwork_t*)malloc(sizeof(nnwork_t));

	ret->inum = inum;
	ret->hnum = hnum;
	ret->onum = onum;
	ret->ihw = (double**)malloc(sizeof(double*) * inum);
	for (i = 0; i < inum; i++)
		ret->ihw[i] = (double*)malloc(sizeof(double) * hnum);
	ret->how = (double**)malloc(sizeof(double*) * hnum);
	for (h = 0; h < hnum; h++)
		ret->how[h] = (double*)malloc(sizeof(double) * onum);
	ret->hout = (double*)malloc(sizeof(double) * hnum);
	ret->lambda = 1.0;
	ret->rate = 0.25;
	ret->mrate = .00005;
	ret->mlow = 0.5;
	ret->mhigh = 2.0;

	for (i = 0; i < inum; i++) for (h = 0; h < hnum; h++)
			ret->ihw[i][h] = randrange(-0.5, 0.5);

	for (h = 0; h < hnum; h++) for (o = 0; o < onum; o++)
			ret->how[h][o] = randrange(-0.5, 0.5);
	
	return ret;
}

// free a nnwork_t
void nnwork_destroy(nnwork_t *n) {
	unsigned long i, h;
	for (i = 0; i < n->inum; i++)
		free(n->ihw[i]);
	free(n->ihw);
	for (h = 0; h < n->hnum; h++)
		free(n->how[h]);
	free(n->how);
	free(n->hout);
}

// run a network on inputs
void nnwork_run(nnwork_t *n, double *input, double *output) {
	double sum;
	int i, h, o;

	for (h = 0; h < n->hnum; h++) {
		sum = 0;

		for (i = 0; i < n->inum; i++) {
			if (n->ihw[i][h] != 0.0)
				sum += n->ihw[i][h] * input[i];
		}

		n->hout[h] = nnwork_sigmoid(sum, n->lambda);
	}

	for (o = 0; o < n->onum; o++) {
		sum = 0;

		for (h = 0; h < n->hnum; h++)
			if (n->how[h][o] != 0.0)
				sum += n->how[h][o] * n->hout[h];

		output[o] = nnwork_sigmoid(sum, n->lambda);
	}
}

// run then adjust a network
void nnwork_train(nnwork_t *n, double *input, double *desired) {
	unsigned int i, h, o;
	double sum;
	double *output = (double*)malloc(sizeof(double) * n->onum);
	double *deltas = (double*)malloc(sizeof(double) * n->onum);
	double **innerdeltas = (double**)malloc(sizeof(double*) * n->hnum);
	for (h = 0; h < n->hnum; h++)
		innerdeltas[h] = (double*)malloc(sizeof(double) * n->onum);

	nnwork_run(n, input, output);

	for (o = 0; o < n->onum; o++)
		deltas[o] = (desired[o]-output[o]) * output[o] * (1.0-output[o]);

	for (o = 0; o < n->onum; o++) for (h = 0; h < n->hnum; h++) {
		if (n->how[h][o] != 0.0)
			innerdeltas[h][o] = n->rate * deltas[o] * n->hout[h];
	}

	for (h = 0; h < n->hnum; h++) for (i = 0; i < n->inum; i++) {
		if (n->ihw[h][o] != 0.0) {
			sum = 0.0;

			for (o = 0; o < n->onum; o++)
				sum += deltas[o] * n->how[h][o];

			n->ihw[i][h] += n->rate * n->hout[h] * (1.0 - n->hout[h]) * sum * input[i];
		}
	}

	for (o = 0; o < n->onum; o++) for (h = 0; h < n->hnum; h++)
		if (n->how[h][o] != 0.0)
			n->how[h][o] += innerdeltas[h][o];

	for (h = 0; h < n->hnum; h++)
		free(innerdeltas[h]);
	free(innerdeltas);
	free(deltas);
	free(output);
}

nnwork_t** nnwork_breed(nnwork_t *l, nnwork_t *r, unsigned short children) {
	unsigned long long i, h, o, c;
	nnwork_t **ret = (nnwork_t**)malloc(children * sizeof(nnwork_t*));
	double *lg = (double*)malloc(sizeof(double) * (l->inum*l->hnum + l->hnum*l->onum));
	double *rg = (double*)malloc(sizeof(double) * (r->inum*r->hnum + r->hnum*r->onum));
	int crossover;
	int len;
	double tmp;
	
	if ((l->inum != r->inum) || (l->hnum != r->hnum) || (l->onum != r->onum))
		return NULL;

	c = 0;
	for (i = 0; i < l->inum; i++) for (h = 0; h < l->hnum; h++)
		lg[c++] = l->ihw[i][h];
	for (h = 0; h < l->hnum; h++) for (o = 0; o < l->onum; o++)
		lg[c++] = l->how[h][o];

	c = 0;
	for (i = 0; i < r->inum; i++) for (h = 0; h < r->hnum; h++)
		rg[c++] = r->ihw[i][h];
	for (h = 0; h < r->hnum; h++) for (o = 0; o < r->onum; o++)
		rg[c++] = r->how[h][o];

	len = c;

	for (c = 0; c < children; c++) {
		ret[c] = nnwork_init(l->inum, l->hnum, l->onum);
		ret[c]->mrate = randrange(l->mrate, r->mrate);
		ret[c]->mlow = randrange(l->mlow, r->mlow);
		ret[c]->mhigh = randrange(l->mhigh, r->mhigh);
		crossover = randrange(1.0, len);

		o = 0;
		for (i = 0; i < l->inum; i++) for (h = 0; h < l->hnum; h++) {
			if (o++ < crossover)
				ret[c]->ihw[i][h] = lg[o];
			else
				ret[c]->ihw[i][h] = rg[o];

			if(randrange(0.0, 1.0) <= ret[c]->mrate)
				ret[c]->ihw[i][h] *= randrange(ret[c]->mlow, ret[c]->mhigh);
		}
		i = o;
		for (h = 0; h < l->hnum; h++) for (o = 0; o < l->onum; o++) {
			if (i++ < crossover) 
				ret[c]->how[h][o] = lg[i];
			else
				ret[c]->how[h][o] = rg[i];

			if(randrange(0.0, 1.0) <= ret[c]->mrate)
				ret[c]->how[h][o] *= randrange(ret[c]->mlow, ret[c]->mhigh);
		}
	}

	return ret;
}
