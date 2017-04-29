/**
 * MPI implementation of a fully-connected feed forward neural network
 *
 * Authors: Jevgenija Aksjonova (jevaks@kth.se)
 *          Beatrice Ionascu (bionascu@kth.se)
 *
 * Last changed: 04/28/2017
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"


#define TOL 0.00000001
#define BATCH_SIZE 100
#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000
#define FUNCTION_TYPE 1 // 0 - tanh, 1 - relu


/* Read labels data */
void read_labels(const char filename[], int *array, int data_size) {
    memset(array, 0, sizeof(int) * data_size);
    FILE *file = fopen(filename, "r");
    if (file) {
        size_t i;
        /* Read each line from the file. */
        for (i = 0; i < data_size && !feof(file); i++) {
            fscanf(file, "%d", array + i);
        }
        assert(i == data_size);
        fclose(file);
    } else {
        perror(filename);
    }
}


/* Read image data */
void read_images(const char *filename, double *array, int data_size, int batch_size, int batch_ind) {
    FILE *file = fopen(filename, "r");
    if (file) {
        size_t i, j;
        char buffer[data_size * 4], *ptr;
        /* Read each line from the file. */
        for (i = 0; fgets(buffer, sizeof buffer, file); ++i) {
            /* Only parse data from desired batch. */
            if (i >= batch_ind * batch_size && i < (batch_ind + 1) * batch_size ) {
                /* Parse the comma-separated values from each line into 'array'. */
                for (j = 0, ptr = buffer; j < data_size; j++, ptr++) {
                    array[(i - batch_ind * batch_size) * data_size + j] = strtol(ptr, &ptr, 10) / 256.0;
                }
            }
        }
        fclose(file);
    } else {
        perror(filename);
    }
}


/* Compute local size of layers */
int local(int layer_size, int p, int P) {
    int I = (layer_size + P - p - 1) / P;
    return I;
}


/* Compute local number of parameters (includes weights and biases) */
int param_size(int *layer_size, int num_layers, int p, int P) {
    int size = 0;
    size_t i;
    for (i = 1; i < num_layers; i++) {
        size = size + local(layer_size[i], p, P) * (layer_size[i-1] + 1);
    }
    return size;
}


/* Compute total amount of data (includes the input data features and the layer outputs) */
int data_size(int *layer_size, int num_layers) {
    int size = 0;
    size_t i;
    for (i = 0; i < num_layers; i++) {
        size += layer_size[i];
    }
    return size;
}


/* Compute local starting index of the parameters corresponding to layer */
int param_ind(int layer, int p, int P, int *layer_size) {
    int ind = 0;
    size_t i;
    for (i = 1; i < layer; i++) {
        ind += local(layer_size[i], p, P) * (layer_size[i - 1] + 1);
    }
    return ind;
}


/* Compute local starting index of the data corresponding to layer */
int data_ind(int layer, int p, int P, int *layer_size) {
    int ind = 0;
    size_t i;
    for (i = 0; i < layer; i++) {
        ind += layer_size[i];
    }
    for (i = 0; i < p; i++) {
        ind += local(layer_size[layer], (int) i, P);
    }
    return ind;
}


/* Forward pass */
void forward(double *input, int inputSize, double *output, int outputSize, double *param, int fun) {
    size_t i, j;
    for (i = 0; i < outputSize; i++) {
        /* Initialize output with the bias term */
        output[i] = param[(inputSize + 1) * i];
        for (j = 0; j < inputSize; j++) {
            // Add weighted inputs
            output[i] += param[(inputSize + 1)* i + j + 1] * input[j];
        }
        /* Activation */
        if (fun > 0) {
            if (FUNCTION_TYPE == 0) {
                output[i] = tanh(output[i]);
            }
            else if (FUNCTION_TYPE == 1) { // relu function
                if (output[i] <= 0) {
                    output[i] = 0.0;
                }
            }
        }
    }
}


/* Backward pass (backpropagate gradients from current_layer to prev_layer) */
void backward(double *current_layer, int local_layer_size, double *prev_layer, int prev_layer_size, double *param,
              double *grad_param, int layer, double lambda) {
    size_t i, j, k;
    /* Backpropagate into the parameters */
    for (i = 0; i < local_layer_size; i++) {
        /* Bias terms - sum gradients */
        grad_param[i * (prev_layer_size + 1)] += current_layer[i];
        /* Weights - use chain rule and sum */
        for (j = 0; j < prev_layer_size; j++) {
            grad_param[i * (prev_layer_size + 1) + j + 1] += current_layer[i] * prev_layer[j] +
                    2.0 * lambda * param[i * (prev_layer_size + 1) + j + 1]; // regularization
        }
    }
    /* Backpropagate into the layer and save local gradients in prev_layer (data array) */
    if (layer > 1) {
        for (i = 0; i < prev_layer_size; i++) {
            if (FUNCTION_TYPE == 0) {
                double localGrad = (1-pow(prev_layer[i],2)); // tanh derivative
                prev_layer[i] = 0.0; 
                for (k = 0; k < local_layer_size; k++) {
                    prev_layer[i] += current_layer[k] * param[k * (prev_layer_size +1) + i + 1];
                }
                prev_layer[i] *= localGrad;
            } else if (FUNCTION_TYPE == 1) {
                if (prev_layer[i] > 0.0) {
                    prev_layer[i] = 0.0;
                    for (k = 0; k < local_layer_size; k++) {
                        prev_layer[i] += current_layer[k] * param[k * (prev_layer_size + 1) + i + 1];
                    }
                } else {
                    prev_layer[i] = 0.0;
                }
            }
        }
    }
}


/* Train the network */
void train(const char filename[], int* label, double *param, double *grad_param, int *layer_size, int num_layers, int p,
           int P, int epochs, double lambda, double learning_rate) {
    size_t img, ep = 0;
    int i, layer, batch_index;
    int input_pointer, output_pointer, param_pointer, probs_pointer;
    double sum, global_loss = 1;
    int image_size = layer_size[0];
    int num_param = param_size(layer_size, num_layers, p, P);

    /* Allocate image data array */
    double *images;
    images = (double *) malloc(BATCH_SIZE * image_size * sizeof(double));

    /* Allocate data array */
    double *data ;
    data = (double *) malloc(data_size(layer_size, num_layers) * sizeof(double));

    /* Loop over epochs (one epoch = loop over all images) */
    while (ep++ < epochs && global_loss > TOL) {
        /* Loop over batches */
        for (batch_index = 0; batch_index < TRAIN_SET_SIZE / BATCH_SIZE; batch_index++) {
            /* Initialize loss */
            global_loss = 0.0;

            /* Reset gradients to 0 */
            memset(grad_param, 0.0, num_param * sizeof(double));

            /* Load images in batch */
            read_images(filename, images, image_size, BATCH_SIZE, batch_index);

            /* Loop over images in batch */
            for (img = 0; img < BATCH_SIZE; img++) {

                /* Copy one image to the 1st layer of the data */
                memcpy(data, images + img * image_size, image_size * sizeof(double));

                /* Forward pass */
                for (layer = 1; layer < num_layers; layer++) {
                    input_pointer = data_ind(layer - 1, 0, P, layer_size);
                    output_pointer = data_ind(layer, p, P, layer_size);
                    param_pointer = param_ind(layer, p, P, layer_size);
                    forward(data + input_pointer, layer_size[layer - 1], data + output_pointer,
                            local(layer_size[layer], p, P), param + param_pointer, num_layers - layer - 1);
                    /* Communicate the data array */
                    double *data_merged;
                    data_merged = (double *) malloc(layer_size[layer] * sizeof(double));
                    int lsizes[P], lpointers[P], lp;
                    for (lp = 0; lp < P; lp++) {
                        lsizes[lp] = local(layer_size[layer], lp, P);
                        if (lp > 0) {
                            lpointers[lp] = lsizes[lp - 1] + lpointers[lp - 1];
                        } else {
                            lpointers[lp] = 0;
                        }
                    }
                    MPI_Allgatherv(data + output_pointer, local(layer_size[layer], p, P), MPI_DOUBLE, data_merged,
                                   lsizes, lpointers, MPI_DOUBLE, MPI_COMM_WORLD);
                    memcpy(data + data_ind(layer, 0, P, layer_size), data_merged, layer_size[layer] * sizeof(double));
                    free(data_merged);
                }

                /* Softmax */
                probs_pointer = data_ind(num_layers - 1, 0, P, layer_size);
                sum = 0.0;
                for ( i = 0; i < layer_size[num_layers - 1]; i++) {
                    sum += exp(data[probs_pointer + i]);
                }
                for ( i = 0; i < layer_size[num_layers - 1]; i++) {
                    data[probs_pointer + i] = exp(data[probs_pointer + i]) / sum;
                }

                /* Loss computation */
                /* Compute the cross-entropy loss by adding the log probabilities assigned to the correct classes */
                global_loss += -log(data[probs_pointer + label[batch_index * BATCH_SIZE + img]]);

                /* Gradient computation for the last (score) layer */
                data[probs_pointer + label[batch_index * BATCH_SIZE + img]] -= 1.0;

                /* Backpropagation */
                for (layer = num_layers - 1; layer > 0; layer--) {
                    input_pointer = data_ind(layer, p, P, layer_size);
                    output_pointer = data_ind(layer - 1, 0, P, layer_size);
                    param_pointer = param_ind(layer, p, P, layer_size);
                    backward(data + input_pointer, local(layer_size[layer], p, P), data + output_pointer,
                             layer_size[layer - 1], param + param_pointer, grad_param + param_pointer, layer, lambda);
                    /* Reduce (sum) gradients */
                    if (layer > 1) {
                        double *data_merged;
                        data_merged = (double *) malloc(layer_size[layer - 1] * sizeof(double));
                        MPI_Allreduce(data + output_pointer, data_merged, layer_size[layer - 1], MPI_DOUBLE, MPI_SUM,
                                      MPI_COMM_WORLD);
                        memcpy(data + output_pointer, data_merged, layer_size[layer - 1] * sizeof(double));
                        free(data_merged);
                    }
                }
            }

            /* Update parameters using gradient averaged over batch */
            for (i = 0; i < num_param; i++) {
                param[i] -= learning_rate * grad_param[i] / (float) BATCH_SIZE;
            }

            /* Compute global loss averaged over batch */
            global_loss /= BATCH_SIZE;
            if (p == 0 && batch_index % 100 == 0) {
                printf("Ep: %zu/%d\t batch:%d \t train loss: %f\n", ep, epochs, batch_index, global_loss);
            }
        }
    }
    free(data);
    free(images);
}


/* Test the network */
void test(const char filename[], int* label, double *param, int *layer_size, int num_layers, int p, int P) {
    int it, layer, i;
    int input_pointer, output_pointer, param_pointer, probs_pointer, predicted_class;
    double sum, predicted_class_score, global_loss = 0.0;
    int image_size = layer_size[0];
    int acc = 0;

    /* Allocate image data array */
    double *images;
    images  = (double *) malloc(image_size* sizeof(double));

    /* Allocate data array */
    double *data = (double *) malloc(data_size(layer_size, num_layers)*sizeof(double));

    /* Loop over batches */
    for (it = 0; it < TEST_SET_SIZE; it++) {

        /* Load image data */
        read_images(filename, images, image_size, 1, it);

        /* Copy one image to the 1st layer of the data */
        memcpy(data, images, image_size * sizeof(double));

        /* Forward pass */
        for (layer = 1; layer < num_layers; layer++) {
            input_pointer = data_ind(layer - 1, 0, P, layer_size);
            output_pointer = data_ind(layer, p, P, layer_size);
            param_pointer = param_ind(layer, p, P, layer_size);
            forward(data + input_pointer, layer_size[layer - 1], data + output_pointer, local(layer_size[layer], p, P),
                    param + param_pointer, num_layers - layer - 1);
            /* Communicate the data array */
            double *data_merged;
            data_merged = (double *) malloc(layer_size[layer] * sizeof(double));
            int lsizes[P], lpointers[P], lp;
            for (lp = 0; lp < P; lp++) {
                lsizes[lp] = local(layer_size[layer], lp, P);
                if (lp > 0) {
                    lpointers[lp] = lsizes[lp - 1] + lpointers[lp - 1];
                } else {
                    lpointers[lp] = 0;
                }
            }
            MPI_Allgatherv(data + output_pointer, local(layer_size[layer], p, P), MPI_DOUBLE, data_merged, lsizes,
                           lpointers, MPI_DOUBLE, MPI_COMM_WORLD);
            memcpy(data + data_ind(layer, 0, P, layer_size), data_merged, layer_size[layer] * sizeof(double));
            free(data_merged);
        }

        /* Softmax and Predict class */
        probs_pointer = data_ind(num_layers - 1, 0, P, layer_size);
        sum = 0.0;
        for (i = 0; i < layer_size[num_layers - 1]; i++) {
            sum += exp(data[probs_pointer + i]);
        }
        predicted_class_score = -1.0;
        predicted_class = -1;
        for (i = 0; i < layer_size[num_layers - 1]; i++) {
            /* Compute normalized probability of the correct class */
            data[probs_pointer + i] = exp(data[probs_pointer + i]) / sum;
            /* Argmax */
            if (data[probs_pointer+i] > predicted_class_score) {
                predicted_class_score = data[probs_pointer + i];
                predicted_class = i;
            }
        }

        /* Compute accuracy */
        if (predicted_class == label[it]) {
            acc++;
        }

        /* Compute the cross-entropy loss by adding the log probabilities assigned to the correct classes */
        global_loss += -log(data[probs_pointer + label[it]]);

        if (p == 0 && (it+1) % 1000 == 0) {
            printf("Test set size: %d\tloss: %f\taccuracy:%f\n",
                   it + 1, global_loss / (it + 1), (float) acc / (float) (it + 1));
        }
    }
    free(data);
    free(images);
}



int main(int argc, char **argv) {

    /* local variables */
    int p, P, epochs, W = 28, H = 28;
    MPI_Status status;
    int tag = 100;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &p);

    /* Read number of epochs from command line */
    if (argc < 2) {
        printf("No batch size given \n");
        exit(1);
    }
    epochs = atoi(argv[1]);

    /* Set neural network parameters */
    double lambda = 0.0005, learning_rate = 0.01;
    int num_layers = 4;
    int layerSize[4] = {W*H, 300, 100, 10};
    double *param, *grad_param;
    int *train_label, *test_label;

    if (p==0) printf("%d-layer network (learning_rate: %f, reg_stregth: %f, epochs: %d\n",
                     num_layers-1, learning_rate, lambda, epochs);

    /* Paths */
    const char train_images_filename[] = "./mnist_data/train_images.csv";
    const char train_labels_filename[] = "./mnist_data/train_labels.csv";
    const char test_images_filename[] = "./mnist_data/test_images.csv";
    const char test_labels_filename[] = "./mnist_data/test_labels.csv";

    /* Allocate memory and read labels */
    train_label = (int *) malloc(TRAIN_SET_SIZE * sizeof(int));
    read_labels(train_labels_filename, train_label, TRAIN_SET_SIZE);
    test_label = (int *) malloc(TEST_SET_SIZE * sizeof(int));
    read_labels(test_labels_filename, test_label, TEST_SET_SIZE);

    /* Allocate memory for parameters and gradients */
    param = (double *) malloc(param_size(layerSize, num_layers, p, P) * sizeof(double));
    grad_param = (double *) malloc(param_size(layerSize, num_layers, p, P) * sizeof(double));

    /* Initialize a (pseudo-) random number generator */
    srandom(p + 1);

    /* Initialize parameters using "xavier" initialization */
    int layer, i, j;
    for (layer = 1; layer < num_layers; layer++) {
        double a = sqrt(3.0/(float)layerSize[layer-1]); // uniform interval limit
        int param_pointer = param_ind(layer, p, P, layerSize);
        for (i = 0; i < local(layerSize[layer], p, P); i++) {
            param[param_pointer + i * (layerSize[layer - 1] + 1)] = 0.0; // bias
            for (j = 1; j < layerSize[layer - 1] +1; j++) {
                param[param_pointer + i * (layerSize[layer - 1] + 1) + j] = 2.0 * a * (double) random() / RAND_MAX - a;
            }
        }
    }

    /* Initialize gradients with 0 */
    memset(grad_param, 0, param_size(layerSize, num_layers, p, P) * sizeof(double));

    /* Train network */
    train(train_images_filename, train_label, param, grad_param, layerSize, num_layers, p, P, epochs, lambda, learning_rate);

    /* Test network */
    test(test_images_filename, test_label, param, layerSize, num_layers, p, P);

    /* Deallocate arrays */
    free(param);
    free(grad_param);

    MPI_Finalize();

    return 0;
}
