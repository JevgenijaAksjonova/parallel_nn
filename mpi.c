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
    local_loss = (1 / batch_size) * -log(probs[label]); // correction (dont do it on each processor)
    // Add the L2
    for (int i = 0; i < num_param; i++) {
        local_loss += lambda * pow(param[i], 2);
    }
    //}
    return local_loss;
}


/* Backward pass and parameter update */
void backward(double *current_layer, int local_layer_size, double *prev_layer, int prev_layer_size,
              double *param, double lambda) {
    // using mini-batch (stochastic) gradient descent

    //param_size = (prev_layer_size +1) * local_layer_size;
    for (int i = 0; i < local_layer_size; i++) {
        grad_param[i*(prev_layer_size +1)] = current_layer[i];
        for (int j = 0; j < prev_layer_size; j++) { // entire next layer
            grad_param[i*(prev_layer_size +1)+j+1] += current_layer[i] * prev_layer[j];
        }    
    }
    for (int i = 0; i < prev_layer_size; i++) {
        double localGrad = (1-pow(prev_layer[i],2));
        prev_layer[i] = 0.0;
        for (int k = 0; k < local_layer_size; k++) {
            prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
        }
        prev_layer[i] *= localGrad; 
    }
/*
    double local_grad, dparam, above_grad = 0;

    for (int i = 0; i < local_layer_size; i++) { // only local layer
        // Compute local gradient (backpropagate the tanh activation function)
        local_grad = 1 - pow(current_layer[i],2);
        // Sum local gradients from above
        for (int j = 0; j < above_layer_size; j++) { // entire next layer
            above_grad += current_layer[i] * above_layer[j];
        }
        // Compute local gradient by chain rule and save it in corresponding node
        current_layer[i] = local_grad * above_grad;
        // Update parameters
        for (int k = 0; k < below_layer_size + 1; k++) { // entire previous layer
            // Add the regularization gradient (unless the parameter is a bias term)
            if (k > 0) {
                dparam = current_layer[i] + 2 * lambda * param[(below_layer_size)* i + k];
            }
            else {
                dparam = current_layer[i];
            }
            // Perform a parameter update
            param[(below_layer_size)* i + k] =+ - learning_rate * dparam;
        }
    }*/
}

void train(double *data, int label, double *param, int *layerSize, int *localLayerSize, int Nlayers, int p, int P, int batch_size, double lambda, double learning_rate) {

    int it, itTotal = 0, layer;
    double local_loss, global_loss = 1;
    while (itTotal++ < MAX_ITER && global_loss > TOL) {

        for (it = 0; it < NUM_IT_PER_TEST; it++) {

            image_size = layerSize[0];

            /* Allocate array */
            image = malloc(image_size); //malloc(batch_size * image_size);
            //for (size_t i = 0; i < N; i++) {
            //    images[i] = malloc(dataSize(layerSize, Nlayers) * sizeof(*data[i]));
            //}
            read_csv(filename, image, W * H, 1, 0);

            double *data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));
            //copy image to the 1st layer of the data            

            /* Forward pass */
            for (layer = 1; layer < Nlayers; layer++) {
                int inputPointer = dataInd(layer-1, 0, P, layerSize);
                int outputPointer = dataInd(layer, p, P, layerSize);
                int paramPointer = paramInd(layer, p, P, layerSize);
                forward(data+inputPointer, layerSize[layer-1], data+outputPointer, local(layerSize[layer],p,P), param+paramPointer, Nlayers-layer-1);
                MPI_Alltoall(data+outputPointer, local(layerSize[layer],p,P), MPI_DOUBLE, data+dataInd(layer+1, p, P, layerSize),);
            }

            /* Loss computation */
            int probs_pointer = dataInd(Nlayers-1, 0, P, layerSize); // correction
            int num_param = paramSize(layerSize, Nlayers, p, P);
            local_loss = compute_loss(data + probs_pointer, param, label, num_param, batch_size, lambda); // use lambda = 0 to ignore regularization
            if (p == 0) {
                MPI_Reduce(&local_loss, &global_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                printf("total loss: ");
            }

            /* Gradient computation for the last (score) layer */
            data[probs_pointer+label] -= 1;
            for (int i = 0; i < layerSize[Nlayers-1]; i++) {
                data[probs_pointer+i] /= batch_size; // todo check lectures why this is necessary
            }

            /* Backpropagation */
            for (layer = Nlayers-1; layer > 0; layer--) {
                int inputPointer = dataInd(layer, p, P, layerSize);
                int outputPointer = dataInd(layer-1, 0, P, layerSize);
                int paramPointer = paramInd(layer, p, P, layerSize);

                int abovePointer = dataInd(layer+1, p, P, layerSize);
                backward(data+inputPointer, local(layerSize[layer],p,P), data+outputPointer, layerSize[layer-1],
                         param+paramPointer, lambda);
                MPI_Reduce(data+outputPointer, layerSize[layer-1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // how to sum up the entire array
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
    grad_param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));

/* Initialize a (pseudo-) random number generator */
    srandom(p + 1);

/* Initialize parameters as small random values */
    for (int i = 0; i < paramSize(layerSize, Nlayers, p, P); i++) {
        param[i] = 0.01 * (double) random() / RAND_MAX;
    }


/* NN */
    train(filename);

    //test();

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
