#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use MPI */
#include "mpi.h"

#define MAX_ITER 1
#define TOL 0.00000001
#define TRAIN_SET_SIZE 60000
#define TEST_SET_SIZE 10000
#define FUNCTION_TYPE 1 // 0 - tanh, 1 - relu


int read_labels(const char filename[], int *array, int data_size) {
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

        return 1;
    } else {
        perror(filename);
        return 0;
    }
}

void read_images(const char *filename, double *array, int data_size, int batch_size, int batch_ind) {
    FILE *file = fopen(filename, "r");
    if (file) {
        size_t i, j;
        char buffer[data_size * 4], *ptr;
        /* Read each line from the file. */
        for (i = 0; fgets(buffer, sizeof buffer, file); ++i) {
            /* Only parse data from desired batch. */
            if (i >= batch_ind*batch_size && i < (batch_ind+1)*batch_size ) {
                /* Parse the comma-separated values from each line into 'array'. */
                for (j = 0, ptr = buffer; j < data_size; j++, ptr++) {
                    array[(i-batch_ind*batch_size)*data_size+j] = strtol(ptr, &ptr, 10)/256.0;
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
            if (FUNCTION_TYPE == 0) {
                output[i] = tanh(output[i]);
            } else if (FUNCTION_TYPE == 1) { // relu function
                if (output[i] <= 0) {
                    output[i] = 0.0;
                }
            }
        }
    }
}


/* Compute L2 regularization term on the parameters */
double compute_regularization_loss(double *param, int num_param, double lambda) {
    double local_loss = 0;
    for (int i = 0; i < num_param; i++) {
        local_loss += pow(param[i], 2);
    }
    return lambda*local_loss;
}


/* Backward pass and parameter update */
void backward(double *current_layer, int local_layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {
    
    // Update parameters
    //param_size = (prev_layer_size +1) * local_layer_size;
    for (int i = 0; i < local_layer_size; i++) {
        grad_param[i*(prev_layer_size +1)] += 2.0*current_layer[i]; // multiply by 2 for faster training/better convergence
        for (int j = 0; j < prev_layer_size; j++) {
            // regularisation added
            grad_param[i*(prev_layer_size +1)+j+1] += current_layer[i] * prev_layer[j] + 2.0*lambda*param[i*(prev_layer_size +1)+j+1];
        }
    }
    if (layer > 1) { // backprapogate partial derivatives
        for (int i = 0; i < prev_layer_size; i++) {
            if (FUNCTION_TYPE == 0) {
                double localGrad = (1-pow(prev_layer[i],2)); // tanh derivative
                prev_layer[i] = 0.0; 
                for (int k = 0; k < local_layer_size; k++) {
                    prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
                }
                prev_layer[i] *= localGrad;
            } else if (FUNCTION_TYPE == 1) {
                if (prev_layer[i] >0.0 ) {
                    prev_layer[i] = 0.0;
                    for (int k = 0; k < local_layer_size; k++) {
                        prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
                    }
                } else {
                    prev_layer[i] = 0.0;
                }
            }
        }
    }
}


void train(const char filename[], int* label, double *param, double *grad_param, int *layerSize, int Nlayers, int p, int P, int batch_size, double lambda, double learning_rate, double momentum) {

    int img, itTotal = 0, layer, batch_index;
    double local_reg_loss, global_reg_loss, global_loss = 1;
    int image_size = layerSize[0];
    int num_param = paramSize(layerSize, Nlayers, p, P);

    /* Allocate image data array */
    double * images;
    images = (double * ) malloc(batch_size *image_size* sizeof(double));

    /* Allocate data array */
    double *data ;
    data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));
    for (int i = num_param-10; i < num_param; i++) {
        printf("%f ",param[i]);
    }
    printf("\n");

    /* Epochs, loop over all images */
    while (itTotal++ < MAX_ITER && global_loss > TOL) {

        /* Loop over batches */
        for (batch_index = 0; batch_index < TRAIN_SET_SIZE/batch_size; batch_index++) {

            global_loss = 0.0;
            global_reg_loss = 0.0;

            /* Re-set gradients to 0*/
            memset(grad_param, 0.0, paramSize(layerSize, Nlayers, p, P) * sizeof(double));

            /* Load images in batch */
            read_images(filename, images, image_size, batch_size, batch_index);

            /* Loop over images in batch */
            for (img = 0; img < batch_size; img++) {

                /* Copy one image to the 1st layer of the data */
                memcpy(data, images+img*image_size, image_size*sizeof(double));
                /*for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        if (images[img*image_size+ row*28+col] > 0) printf("o");
                        else printf("-");
                    }
                    printf("\n");
                }
                printf("Label = %d \n",label[batch_index * batch_size + img]); */

                /* Forward pass */
                for (layer = 1; layer < Nlayers; layer++) {
                    int inputPointer = dataInd(layer - 1, 0, P, layerSize);
                    int outputPointer = dataInd(layer, p, P, layerSize);
                    int paramPointer = paramInd(layer, p, P, layerSize);
                    forward(data + inputPointer, layerSize[layer - 1], data + outputPointer, local(layerSize[layer], p, P),
                            param + paramPointer, Nlayers - layer - 1);
                    /* Communiate the data array */
                    double *dataMerged;
                    dataMerged = (double *) malloc(layerSize[layer] * sizeof(double));
                    int lsizes[P], lpointers[P];
                    for (int lp = 0; lp < P; lp++) {
                        lsizes[lp] = local(layerSize[layer], lp, P);
                        if (lp > 0) {
                            lpointers[lp] = lsizes[lp - 1] + lpointers[lp - 1];
                        } else {
                            lpointers[lp] = 0;
                        }
                    }
                    int err = MPI_Allgatherv(data + outputPointer, local(layerSize[layer], p, P), MPI_DOUBLE, dataMerged,
                                             lsizes, lpointers, MPI_DOUBLE, MPI_COMM_WORLD);
                    memcpy(data + dataInd(layer, 0, P, layerSize), dataMerged, layerSize[layer] * sizeof(double));
                    free(dataMerged);
                }

                /* Softmax */
                int probs_pointer = dataInd(Nlayers - 1, 0, P, layerSize);
                double sum = 0.0;
                for (int i = 0; i < layerSize[Nlayers - 1]; i++) {
                    sum += exp(data[probs_pointer + i]);
                }
                for (int i = 0; i < layerSize[Nlayers - 1]; i++) {
                    data[probs_pointer + i] = exp(data[probs_pointer + i]) / sum;
                }

                /* Loss computation */
                /* Compute the cross-entropy loss */
                /* Add the log probabilities assigned to the correct classes */
                global_loss += -log(data[probs_pointer + label[batch_index * batch_size + img]]);

                /* Gradient computation for the last (score) layer */
                data[probs_pointer + label[batch_index * batch_size + img]] -= 1.0;

                /* Add the regularization loss */
                //local_reg_loss = compute_regularization_loss(param, num_param, lambda); // use lambda = 0 to ignore regularization
                //if (p == 0) {
                    //MPI_Reduce(&local_reg_loss, &global_reg_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    //global_loss += local_reg_loss;
                //}


                /* Backpropagation */
                for (layer = Nlayers - 1; layer > 0; layer--) {
                    int inputPointer = dataInd(layer, p, P, layerSize);
                    int outputPointer = dataInd(layer - 1, 0, P, layerSize);
                    int paramPointer = paramInd(layer, p, P, layerSize);

                    backward(data + inputPointer, local(layerSize[layer], p, P), data + outputPointer, layerSize[layer - 1],
                             param + paramPointer, grad_param + paramPointer, layer, lambda);
                    if (layer > 1) {
                        double *dataMerged;
                        dataMerged = (double *) malloc(layerSize[layer - 1] * sizeof(double));
                        int res = MPI_Allreduce(data + outputPointer, dataMerged, layerSize[layer - 1], MPI_DOUBLE, MPI_SUM,
                                                MPI_COMM_WORLD);
                        memcpy(data + outputPointer, dataMerged, layerSize[layer - 1] * sizeof(double));
                        free(dataMerged);
                    }
                }
            }

            /* Update parameters using gradient averaged over batch)*/
            for (int i = 0; i < num_param; i++) {
                param[i] = param[i] - learning_rate * grad_param[i] / (float)batch_size;
            }

            /* Compute global loss averaged over batch */
            global_loss /= batch_size;
            if (p==0 && batch_index % 100 == 0) { 
                printf("Iteration - %d, batch - %d, total loss: %f\n", itTotal, batch_index, global_loss); 
            }
        }
    }
    for (int i = num_param-10; i < num_param; i++) {
        printf("%f ",param[i]);
    }
    printf("\n");
    free(data);
    free(images);
}

void test(const char filename[], int* label, double *param, int *layerSize, int Nlayers, int p, int P) {

    int it, layer;
    double local_reg_loss, global_reg_loss = 0.0, global_loss = 0.0;
    int image_size = layerSize[0];
    int acc = 0;

    /* Allocate image data array */
    double *images;
    images  = (double *) malloc(image_size* sizeof(double));

    /* Allocate data array */
    double *data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));

    /* Loop over batches */
    for (it = 0; it < TEST_SET_SIZE; it++) {

        /* Load image data */
        read_images(filename, images, image_size, 1, it);

        /* Copy one image to the 1st layer of the data */
        memcpy(data, images, image_size*sizeof(double));
        /*for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                if (data[row*28+col] > 0) printf("o");
                else printf("-");
            }
            printf("\n");
        }
        printf("Label = %d \n",label[it]); */

        /* Forward pass */
        for (layer = 1; layer < Nlayers; layer++) {
            int inputPointer = dataInd(layer - 1, 0, P, layerSize);
            int outputPointer = dataInd(layer, p, P, layerSize);
            int paramPointer = paramInd(layer, p, P, layerSize);
            forward(data + inputPointer, layerSize[layer - 1], data + outputPointer, local(layerSize[layer], p, P),
                    param + paramPointer, Nlayers - layer - 1);
            /* Communiate the data array */
            double *dataMerged;
            dataMerged = (double *) malloc(layerSize[layer] * sizeof(double));
            int lsizes[P], lpointers[P];
            for (int lp = 0; lp < P; lp++) {
                lsizes[lp] = local(layerSize[layer], lp, P);
                if (lp > 0) {
                    lpointers[lp] = lsizes[lp - 1] + lpointers[lp - 1];
                } else {
                    lpointers[lp] = 0;
                }
            }
            int err = MPI_Allgatherv(data + outputPointer, local(layerSize[layer], p, P), MPI_DOUBLE, dataMerged,
                                     lsizes, lpointers, MPI_DOUBLE, MPI_COMM_WORLD);
            memcpy(data + dataInd(layer, 0, P, layerSize), dataMerged, layerSize[layer] * sizeof(double));
            free(dataMerged);
        }

        /* Softmax */
        int probs_pointer = dataInd(Nlayers-1, 0, P, layerSize);
        double sum = 0.0;

        for (int i = 0; i < layerSize[Nlayers-1]; i++) {
            sum += exp(data[probs_pointer+i]);
        }
        double predicted_class_score = -1.0;
        int predicted_class = -1;
        for (int i = 0; i < layerSize[Nlayers-1]; i++) {
            /* Compute normalized probability of the correct class */
            data[probs_pointer+i] = exp(data[probs_pointer+i])/sum;
            /* Argmax */
            if (data[probs_pointer+i] > predicted_class_score) {
                predicted_class_score = data[probs_pointer+i];
                predicted_class = i;
            }
        }

        if (predicted_class == label[it]) {
            acc++;
        }

        /* Loss computation */
        int num_param = paramSize(layerSize, Nlayers, p, P);

        /* Compute the cross-entropy loss */

        global_reg_loss = 0;
        /* Add the log probabilities assigned to the correct classes */
        global_loss += -log(data[probs_pointer + label[it]]);

        /* Add the regularization loss */
        //local_reg_loss = compute_regularization_loss(param, num_param, 0.0005); // use lambda = 0 to ignore regularization
        //if (p == 0) {
            //MPI_Reduce(&local_reg_loss, &global_reg_loss, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            //global_loss += local_reg_loss;
        //}

        if (p == 0 && (it+1) % 1000 == 0) {
            printf("TEST set size: %d\n", it+1);
            printf("TEST total loss: %f\n", global_loss / (it+1));
            printf("TEST accuracy: %f\n", (float)acc / (float)(it+1));
        }

    }

    free(data);
    free(images);
}

int main(int argc, char **argv) {

/* local variables */
    int p, P;
    int batch_size, W = 28, H = 28;
    MPI_Status status;
    int tag = 100;
    const char train_images_filename[] = "./mnist_data/train_images.csv";
    const char train_labels_filename[] = "./mnist_data/train_labels.csv";
    const char test_images_filename[] = "./mnist_data/test_images.csv";
    const char test_labels_filename[] = "./mnist_data/test_labels.csv";


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

    if (p==0) printf("batch_Size: %d\n", batch_size);

    /* Set neural network parameters */
    double lambda = 0.0005, learning_rate = 0.01, momentum = 0.9;
    int Nlayers = 4;
    int layerSize[4] = {W*H,300,100,10};
    double *param, *grad_param;
    int *train_label;
    int *test_label;

    /* Allocate memory and read labels */
    train_label = (int *) malloc(TRAIN_SET_SIZE * sizeof(int));
    read_labels(train_labels_filename, train_label, TRAIN_SET_SIZE);

    test_label = (int *) malloc(TEST_SET_SIZE * sizeof(int));
    read_labels(test_labels_filename, test_label, TEST_SET_SIZE);


    /* Allocate memory for parameters */
    param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));
    grad_param = (double *) malloc(paramSize(layerSize, Nlayers, p, P) * sizeof(double));

    /* Initialize a (pseudo-) random number generator */
    srandom(p + 1);

    /* Initialize parameters using "xavier" initialization */
    for (int layer = 1; layer < Nlayers; layer++) {
        double a = sqrt(3.0/(float)layerSize[layer-1]);
        int param_pointer = paramInd(layer, p, P, layerSize);
        for (int i = 0; i < local(layerSize[layer],p,P); i++) {
            param[param_pointer+i*(layerSize[layer-1] +1)] = 0.0; //bias
            for (int j = 1; j < layerSize[layer-1] +1; j++) {
                param[param_pointer+i*(layerSize[layer-1] +1)+j] = 2.0*a * (double) random() / RAND_MAX - a;
            }
        }
    }
    //for (int i = 0; i < paramSize(layerSize, Nlayers, p, P); i++) {
    //    param[i] = 0.01 * (double) random() / RAND_MAX;
    //}

    /* Initialize gradients with 0 */
    memset(grad_param, 0, paramSize(layerSize, Nlayers, p, P) * sizeof(double));

    /* NN */
    /*FILE *f;
    f = fopen("param.txt", "r");
    int num_param = paramSize(layerSize, Nlayers, p, P);
    for (int i = 0; i < num_param; i++) {
        fscanf(f, "%lf", param+i);
    }
    fclose(f);*/
    train(train_images_filename, train_label, param, grad_param, layerSize, Nlayers, p, P, batch_size, lambda, learning_rate, momentum);
    test(test_images_filename, test_label, param, layerSize, Nlayers, p, P);

    MPI_Finalize();

    /* Deallocate arrays */
    free(param);
    free(grad_param);

    return 0;
}
