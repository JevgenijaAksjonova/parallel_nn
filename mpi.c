#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Use MPI */
#include "mpi.h"

#define MAX_ITER = 10000
#define TOL = 0.00000001
#define NUM_IT_PER_TEST = 100

/* compute local size of layers */
int local(int N, int p, int P) {
    int I = (layerSize + P - p - 1)/P;
    return I;
}

/* compute number of parameters */
int paramSize(int *layerSize, int n) {
    int size = 0;
    int i;
    for (i = 1; i < n, i++) {
        size = size+ local(layerSize[i],p,P)*(layerSize[i-1] +1);
    }
    return size;
}

/* compute amount of data */
int dataSize(int *layerSize, int n) {
    int size = 0;
    int i;
    for (i = 0; i < n, i++) {
        size = size+ layerSize[i];
    }
    return size;
}

int paramInd(int layer, int p, int P, int *layerSize, int Nlayers) {
    int i;
    int ind = 0;
    for (i = 1; i < layer; i++) {
        ind = ind + local(layerSize[i],p,P)*(layerSize[i-1] +1);
    }
    return ind;
}

int dataInd(int layer, int p, int P, int *layerSize, int Nlayers) {
    int i;
    int ind = 0;
    for (i = 0; i < layer; i++) {
        ind = ind + layerSize[i];
    }
    for (i = 0; i < p; i++) {
        ind = ind + local(layerSize[layer],p,P);
    }
    return ind;
}

void forward(double *input, int inputSize, double *output, int outputSize, double *param) {
    // number of parameters = (inputSize +1) * outputSize;
}

void backward() {

}

void train(int *layerSize, int * localLayerSize, int Nlayers, int p, int P) {
    int it, itTotal = 0, layer;
    double error;
    while (itTotal++ < MAX_ITER && error > TOL ) {
        for (it = 0; it < NUM_ITER_PER_TEST ; it++) {
            for (layer = 1; layer < Nlayers; layer++) {
                int inputPointer = dataInd(layer-1, 0, P, layerSize, Nlayers);
                int outputPointer = dataInd(layer, p, P, layerSize, Nlayers);
                int paramPointer = paramInd(layer, p, P, layerSize, Nlayers);
                forward(data+inputPointer, layerSize[layer-1], data+outputPointer, local(layerSize[layer],p,P), param+paramPointer);
                MPI_ALL_TO_ALL(data[outputPointer], local(layerSize[layer],p,P));
            }
            softmax();
            for (layer = 1; layer < 3; layer++) {
                backward(layer);
                MPI_ALL_TO_ALL(data[layer], data[]);
            }
        }
        error = test();
    }
}

void test() {

}

int main(int argc, char **argv) {

/* local variables */
    int p, P;
    int N, W = 20, H = 20;
    double *param, *data;
    MPI_Status status;
    int tag = 100;

/* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &p);

/* Read batch size N from command line */
    if (argc < 2) {
        printf("No batch size N given \n");
        exit(1);
    }
    N = atoi(argv[1]);

/* local size */
    int Nlayers = 4;
    int layerSize = {W*H,50*50,50*50,10};
    param = (double *) malloc(paramSize(layerSize,Nlayers)*sizeof(double));
    data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));



/* NN */
    train();

    test();

/* print results */
    //for (i = 0; i < I; i++) {
    //    printf("process - %d; x = %.8f \n", p,x[i]);
    //}

    MPI_Finalize();
    exit(0);
}
