#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use MPI */
#include "mpi.h"

#define MAX_ITER 30
#define TOL 0.00000001
#define NUM_IT_PER_TEST 20


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
                    array[i-batch_ind][j] = strtol(ptr, &ptr, 10);
                }
            }
        }
        fclose(file);
    }
    else {
        perror(filename);
    }
}

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
        ind = ind + local(layerSize[layer],i,P);
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
}


/* Compute L2 regularization term on the parameters */
double compute_regularization_loss(double *param, int num_param, double lambda) {
    double local_loss = 0;
    for (int i = 0; i < num_param; i++) {
        local_loss += lambda * pow(param[i], 2);
    }
    return local_loss;
}


/* Backward pass and parameter update */
void backward(double *current_layer, int local_layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {
    // using mini-batch (stochastic) gradient descent
    // todo add regularization

    //param_size = (prev_layer_size +1) * local_layer_size;
    for (int i = 0; i < local_layer_size; i++) {
        grad_param[i*(prev_layer_size +1)] = current_layer[i];
        for (int j = 0; j < prev_layer_size; j++) { // entire next layer
            grad_param[i*(prev_layer_size +1)+j+1] += current_layer[i] * prev_layer[j];
        }    
    } 
    if (layer > 1) {
        for (int i = 0; i < prev_layer_size; i++) {
            double localGrad = (1-pow(prev_layer[i],2));
            prev_layer[i] = 0.0;
            for (int k = 0; k < local_layer_size; k++) {
                prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
            }
            prev_layer[i] *= localGrad; 
        } 
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


void train(const char filename[], int label, double *param, double *grad_param, int *layerSize, int Nlayers, int p, int P, int batch_size, double lambda, double learning_rate) {

    int it, itTotal = 0, layer;
    double local_reg_loss, global_reg_loss, global_loss = 1;
    int image_size = layerSize[0];

    /* Allocate image data array */
    double **images = malloc(batch_size * sizeof(*images));
    for (size_t i = 0; i < batch_size; i++) {
        images[i] = malloc(image_size * sizeof(*images[i]));
    }

    /* Allocate data array */
    double *data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));

    while (itTotal++ < MAX_ITER && global_loss > TOL) {

        /* Loop over batches */
        for (it = 0; it < NUM_IT_PER_TEST; it++) {

            /* Load image data */
            read_csv(filename, images, image_size, batch_size, it);

            /* Copy one image to the 1st layer of the data */
            memcpy(data, images[0], image_size);

            /* Forward pass */
            for (layer = 1; layer < Nlayers; layer++) {
                int inputPointer = dataInd(layer-1, 0, P, layerSize);
                int outputPointer = dataInd(layer, p, P, layerSize);
                int paramPointer = paramInd(layer, p, P, layerSize);
                forward(data+inputPointer, layerSize[layer-1], data+outputPointer, local(layerSize[layer],p,P), param+paramPointer, Nlayers-layer-1);
                //MPI_Alltoall(data+outputPointer, local(layerSize[layer],p,P), MPI_DOUBLE, data+dataInd(layer+1, p, P, layerSize),);
                double * dataMerged;
                dataMerged = (double *) malloc(layerSize[layer]*sizeof(double));
                int lsizes[P],lpointers[P];
                for (int lp = 0; lp <P; lp++) {
                    lsizes[lp] = local(layerSize[layer],lp,P);
                    if (lp >0 ) {
                        lpointers[lp] = lsizes[lp-1]+lpointers[lp-1];
                    } else {
                        lpointers[lp] = 0;
                    }
                }
                int err = MPI_Allgatherv (data + outputPointer, local(layerSize[layer],p,P), MPI_DOUBLE,dataMerged, lsizes, lpointers,MPI_DOUBLE, MPI_COMM_WORLD) ;
                memcpy( data +dataInd(layer, 0, P, layerSize), dataMerged, layerSize[layer]*sizeof(double));
                free(dataMerged);
            }

            /* Softmax */
            int probs_pointer = dataInd(Nlayers-1, 0, P, layerSize);
            double sum = 0.0;

            for (int i = 0; i < layerSize[Nlayers-1]; i++) {
                sum += exp(data[probs_pointer+i]);
            }
            for (int i = 0; i < layerSize[Nlayers-1]; i++) {
                /* Compute normalized probability of the correct class */
                data[probs_pointer+i] = exp(data[probs_pointer+i])/sum;
            }

            /* Loss computation */
            int num_param = paramSize(layerSize, Nlayers, p, P);

            /* Compute the cross-entropy loss */
            global_loss = 0;
            global_reg_loss = 0;
            for (int i = 0 ; i < batch_size; i++) {
                /* Add the log probabilities assigned to the correct classes */
                global_loss += -log(data[probs_pointer + label]);
            }
            global_loss /= batch_size;

            /* Add the regularization loss */
            //local_reg_loss = compute_regularization_loss(param, num_param, lambda); // use lambda = 0 to ignore regularization
            if (p == 0) {
                //MPI_Reduce(&local_reg_loss, &global_reg_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                global_loss += global_reg_loss;
                printf("total loss: %f\n", global_loss);
            }

            /* Gradient computation for the last (score) layer */
            data[probs_pointer+label] -= 1;
            for (int i = 0; i < layerSize[Nlayers-1]; i++) {
                data[probs_pointer+i] /= batch_size;
                //printf("%3f ", data[probs_pointer+i]);
            }

            /* Backpropagation */
            for (layer = Nlayers-1; layer > 0; layer--) {
                int inputPointer = dataInd(layer, p, P, layerSize);
                int outputPointer = dataInd(layer-1, 0, P, layerSize);
                int paramPointer = paramInd(layer, p, P, layerSize);

                backward(data+inputPointer, local(layerSize[layer],p,P), data+outputPointer, layerSize[layer-1],
                         param+paramPointer, grad_param+paramPointer, layer,lambda);
                if (layer > 1) {
                    double * dataMerged;
                    dataMerged = (double *) malloc(layerSize[layer-1]*sizeof(double));
                    int res = MPI_Allreduce(data+outputPointer,dataMerged,layerSize[layer-1],MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
                    memcpy( data +dataInd(layer-1, 0, P, layerSize), dataMerged, layerSize[layer-1]*sizeof(double));
                    free(dataMerged);
                    //MPI_Reduce(data+outputPointer, layerSize[layer-1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // how to sum up the entire array
                }
            }

            /* Update parameters */
            for (int i = 0; i < num_param; i++) {
                param[i] -= learning_rate * grad_param[i];
            }


        }
    }
    free(data);
    for (size_t i = 0; i < batch_size; i++) {
        free(images[i]);
    }
    free(images);
}

//void test() {
//
//}


int main(int argc, char **argv) {

/* local variables */
    int p, P;
    int batch_size, W = 28, H = 28;
    MPI_Status status;
    int tag = 100;
    const char filename[] = "/Users/nyuad/Documents/workspaceC++/parallel/parallel_nn/mnist_data/train_images.csv";


/* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &p);

/* Read batch size from command line */
    if (argc < 2) {
        printf("No batch size given \n");
        exit(1);
    }
    batch_size = atoi(argv[1]);

    printf("batch_Size: %d\n", batch_size);

    /* Set neural network parameters */
    double lambda = 0, learning_rate = 0.001;
    int Nlayers = 4;
    int layerSize[4] = {W*H,5*4,5*3,10};
    double *param, *grad_param;
    int label = 0; //int *labels;

    /* Allocate memory for parameters */
    param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));
    grad_param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));

    /* Initialize a (pseudo-) random number generator */
    srandom(p + 1);

    /* Initialize parameters as small random values (mean = 0, s.d. = 0.01)*/
    for (int i = 0; i < paramSize(layerSize, Nlayers, p, P); i++) {
        param[i] = 0.01 * (double) random() / RAND_MAX;
    }

    /* Initialize gradients with 0 */
    memset(grad_param, 0, paramSize(layerSize, Nlayers, p, P) * sizeof(double));

    /* NN */
    train(filename, label, param, grad_param, layerSize, Nlayers, p, P, batch_size, lambda, learning_rate);

    //test();

/* print results */
    //for (i = 0; i < I; i++) {
    //    printf("process - %d; x = %.8f \n", p,x[i]);
    //}

    MPI_Finalize();

    /* Deallocate arrays */
    free(param);
    free(grad_param);

    return 0;
}
