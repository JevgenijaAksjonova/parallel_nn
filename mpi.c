#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use MPI */
#include "mpi.h"

#define MAX_ITER 10000
#define TOL 0.00000001
#define NUM_IT_PER_TEST 100

/* compute local size of layers */
int local(int layerSize, int p, int P) {
    int I = (layerSize + P - p - 1)/P;
    return I;
}

/* compute number of parameters */
int paramSize(int *layerSize, int Nlayers, int p, int P) {
    int size = 0;
    int i;
    for (i = 1; i < Nlayers; i++) {
        size = size + local(layerSize[i],p,P)*(layerSize[i-1] +1);
    }
    return size;
}

/* compute amount of data */
int dataSize(int *layerSize, int n) {
    int size = 0;
    int i;
    for (i = 0; i < n; i++) {
        size = size + layerSize[i];
    }
    return size;
}

int paramInd(int layer, int p, int P, int *layerSize) {
    int i;
    int ind = 0;
    for (i = 1; i < layer; i++) {
        ind = ind + local(layerSize[i],p,P)*(layerSize[i-1] +1);
    }
    return ind;
}

int dataInd(int layer, int p, int P, int *layerSize) {
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


/* Forward pass */
void forward(double *input, int inputSize, double *output, int outputSize, double *param, int fun) {
    // number of parameters = (inputSize +1) * outputSize;

    for (int i = 0; i < outputSize; i++) {
        // Initialize output with the bias term
        output[i] = param[(inputSize +1)* i]; // todo check this is correct
        for (int j = 0; j < inputSize; j++) {
            // Add weighted inputs
            output[i] += param[(inputSize +1)* i + j+1] * input[j];
        }
        if (fun > 0) {
            output[i] = tanh(output[i]);
        } 
    }
    
    /* softmax for the last layer */
    if (fun == 0) {
        double sum = 0.0;
        for (int i = 0; i < outputSize; i++) { 
            sum += exp(output[i]);
        }
        for (int i = 0; i < outputSize; i++) {
            /* Compute normalized probability of the correct class */
            output[i] = exp(output[i])/sum;

        }
    }
    
}

/* Compute the cross-entropy loss plus a L2 regularization term on the parameters */
double compute_loss(double *probs, double *param, int label, int num_param, int batch_size, double lambda) { // int *labels
    double local_loss = 0;
    /* Add the log probabilities assigned to the correct classes */
    //for (int i = 0 ; i < batch_size; i++) {
    local_loss = (1 / batch_size) * -log(probs[label]);
    // Add the L2
    for (int i = 0; i < num_param; i++) {
        local_loss += lambda * pow(param[i], 2);
    }
    //}
    return local_loss;
}


/* Backward pass and parameter update */
void backward(double *input, int inputSize, double *output, int outputSize, int fun, int learning_rate) {
    // using mini-batch (stochastic) gradient descent

    double local_grad[outputSize] = {0};

    // Backprop the weighted sum // todo need to access weights from other processes

    // for ..
        // for ..
            // local_grad[] = output[] * input[]


    // Backprop the tanh activation function (everywhere except last layer)
    if (fun != 0) {
        for (int i = 0; i < outputSize; i++) {
            local_grad[i] = 1 - pow(tanh(local_grad[i]),2);
        }
    }

    // Update parameters
    for (int i = 0; i < outputSize; i++) {
        output[i] += - learning_rate * local_grad[i];
    }
}

void train(double *data, int label, double *param, int *layerSize, int *localLayerSize, int Nlayers, int p, int P, int batch_size, double lambda, double learning_rate) {
    int it, itTotal = 0, layer;
    double local_loss, global_loss = 1;
    while (itTotal++ < MAX_ITER && global_loss > TOL) {

        for (it = 0; it < NUM_IT_PER_TEST; it++) {
            /* Forward pass */
            for (layer = 1; layer < Nlayers; layer++) {
                int inputPointer = dataInd(layer-1, 0, P, layerSize);
                int outputPointer = dataInd(layer, p, P, layerSize);
                int paramPointer = paramInd(layer, p, P, layerSize);
                forward(data+inputPointer, layerSize[layer-1], data+outputPointer, local(layerSize[layer],p,P), param+paramPointer, Nlayers-layer-1);
                MPI_Alltoall(data+outputPointer, local(layerSize[layer],p,P), MPI_DOUBLE, data+dataInd(layer+1, p, P, layerSize),);
            }

            /* Loss computation */
            int probs_pointer = dataInd(Nlayers-1, p, P, layerSize);
            int num_param = paramSize(layerSize, Nlayers, p, P);
            local_loss = compute_loss(data + probs_pointer, param, label, num_param, batch_size, lambda); // use lambda = 0 to ignore regularization
            if (p == 0) {
                MPI_Reduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                printf("total loss: ");
            }

            /* Gradient computation for the last (score) layer */
            double grad[10] = {0};
            grad[label] -= 1;
            for (int i = 0; i < 10; i++) {
                grad[i] += data[dataSize(layerSize, Nlayers)-(10-i)];
                grad[i] /= batch_size;
            }

            /* Backpropagation */

            // todo need to access weights from other processes

            backward(grad,layerSize[Nlayers-1], param+paramInd(Nlayers-2, p, P, layerSize), local(layerSize[layer],p,P),Nlayers-layer-1, learning_rate);

            for (layer = Nlayers-2; layer > 0; layer--) {
                int inputPointer = paramInd(layer, 0, P, layerSize); // start index of backprop gradients
                int outputPointer = paramInd(layer-1, p, P, layerSize); // start index of current gradients (to be updated)
                backward(param+inputPointer,layerSize[layer], param+outputPointer, local(layerSize[layer-1],p,P), Nlayers-layer-1, learning_rate);
                //MPI_Alltoall(data[layer], data[]); ???
            }
        }
    }
}

void test() {

}

void read_csv(const char filename[], double **array, int data_size, int batch_size, int batch_ind) {
    FILE *file = fopen(filename, "r");
    if (file) {
        size_t i, j;
        char buffer[data_size * 4], *ptr;
        /* Read each line from the file. */
        for (i = 0; fgets(buffer, sizeof buffer, file); ++i) {
            /* Only parse data from desired batch. */
            if (i >= batch_ind && i < batch_ind + batch_size) {
                /* Parse the comma-separated values from each line into 'array'. */
                for (j = 0, ptr = buffer; j < data_size; j++, ptr++) {
                    array[i][j] = strtol(ptr, &ptr, 10);
                }
            }
        }
        fclose(file);
    }
    else {
        perror(filename);
    }
}

int main(int argc, char **argv) {

/* local variables */
    int p, P;
    int N, W = 20, H = 20;
    MPI_Status status;
    int tag = 100;
    const char filename[] = "/path/train_images.csv";


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

/* Set neural network parameters */
    double learning_rate = 0.0001;
    int batch_index = 0;
    int Nlayers = 4;
    int layerSize[4] = {W*H,50*50,50*50,10};
    double *param;// *grad;
    int label; //int *labels;
    double **data;

/* Allocate memory for parameters */
    param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));
    //grad = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));

/* Initialize a (pseudo-) random number generator */
    srandom(p + 1);

/* Initialize parameters as small random values */
    for (int i = 0; i < paramSize(layerSize, Nlayers, p, P); i++) {
        param[i] = 0.01 * (double) random() / RAND_MAX;
    }

    //data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));

/* Allocate array */
    data = malloc(N + sizeof(*data));
    for (size_t i = 0; i < N; i++) {
        data[i] = malloc(dataSize(layerSize, Nlayers) * sizeof(*data[i]));
    }
    read_csv(filename, data, W * H, N, batch_index);



/* NN */
    train();

    test();

/* print results */
    //for (i = 0; i < I; i++) {
    //    printf("process - %d; x = %.8f \n", p,x[i]);
    //}

    MPI_Finalize();

    /* Deallocate arrays */
    free(param);
    for (size_t i = 0; i < N; i++)
    {
        free(data[i]);
    }
    free(data);

    exit(0);
}
