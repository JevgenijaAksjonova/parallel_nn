#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* compute number of parameters */
int paramSize(int *layerSize, int Nlayers) {
    int size = 0;
    int i;
    for (i = 1; i < Nlayers; i++) {
        size = size + layerSize[i]*(layerSize[i-1] +1);
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

int paramInd(int layer, int *layerSize) {
    int i;
    int ind = 0;
    for (i = 1; i < layer; i++) {
        ind = ind + layerSize[i]*(layerSize[i-1] +1);
    }
    return ind;
}

int dataInd(int layer, int *layerSize) {
    int i;
    int ind = 0;
    for (i = 0; i < layer; i++) {
        ind = ind + layerSize[i];
    }
    return ind;
}


/* Forward pass */
__global__
void forward(double *input, int inputSize, double *output, int outputSize, double *param, int fun) {
    int i = blockIdx.x;

    // Initialize output with the bias term
    output[i] = param[(inputSize +1)* i]; 
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


/* Compute L2 regularization term on the parameters */
double compute_regularization_loss(double *param, int num_param, double lambda) {
    double local_loss = 0;
    for (int i = 0; i < num_param; i++) {
        local_loss += pow(param[i], 2);
    }
    return lambda*local_loss;
}


/* Backward pass and parameter update */
__global__
void backward_part1(double *current_layer, int layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {
    
    int i = blockIdx.x;
    // Update parameters
    grad_param[i*(prev_layer_size +1)] += 2.0*current_layer[i]; // multiply by 2 for faster training/better convergence
    for (int j = 0; j < prev_layer_size; j++) {
        // regularisation added
        grad_param[i*(prev_layer_size +1)+j+1] += current_layer[i] * prev_layer[j] + 2.0*lambda*param[i*(prev_layer_size +1)+j+1];
    }
}

__global__
void backward_part2(double *current_layer, int layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {
    
    int i = blockIdx.x;
    // backprapogate partial derivatives
    if (FUNCTION_TYPE == 0) {
        double localGrad = (1-pow(prev_layer[i],2)); // tanh derivative
        prev_layer[i] = 0.0; 
        for (int k = 0; k < layer_size; k++) {
            prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
        }
        prev_layer[i] *= localGrad;
    } else if (FUNCTION_TYPE == 1) {
        if (prev_layer[i] >0.0 ) {
            prev_layer[i] = 0.0;
            for (int k = 0; k < layer_size; k++) {
                prev_layer[i] += current_layer[k]*param[k*(prev_layer_size +1)+i+1];
            }
        } else {
            prev_layer[i] = 0.0;
        }
    }
}

__global__
void update_param(double *param, double *grad_param, int num_param, double learning_rate, int batch_size) {
    int i = blockIdx.x;
    param[i] = param[i] - learning_rate * grad_param[i] / (float)batch_size;
}


void train(const char filename[], int* label, double *d_param, double *d_grad_param, int *layerSize, int Nlayers, int batch_size, double lambda, double learning_rate, double momentum) {

    int img, itTotal = 0, layer, batch_index;
    double global_loss = 1;
    int image_size = layerSize[0];
    int num_param = paramSize(layerSize, Nlayers);

    /* Allocate image data array */
    double * images;
    images = (double * ) malloc(batch_size *image_size* sizeof(double));

    /* Allocate data array */
    double *d_data ;
    cudaMalloc( (void**)&d_data, dataSize(layerSize,Nlayers)*sizeof(double));
    //data = (double *) malloc(dataSize(layerSize,Nlayers)*sizeof(double));

    /* Epochs, loop over all images */
    while (itTotal++ < MAX_ITER && global_loss > TOL) {

        /* Loop over batches */
        for (batch_index = 0; batch_index < TRAIN_SET_SIZE/batch_size; batch_index++) {

            global_loss = 0.0;

            /* Re-set gradients to 0*/
            cudaMemset(d_grad_param, 0.0, paramSize(layerSize, Nlayers) * sizeof(double));

            /* Load images in batch */
            read_images(filename, images, image_size, batch_size, batch_index);

            /* Loop over images in batch */
            for (img = 0; img < batch_size; img++) {

                /* Copy one image to the 1st layer of the data */
                cudaMemcpy(d_data, images+img*image_size, image_size*sizeof(double), cudaMemcpyHostToDevice );

                /* Forward pass */
                for (layer = 1; layer < Nlayers; layer++) {
                    int inputPointer = dataInd(layer - 1, layerSize);
                    int outputPointer = dataInd(layer, layerSize);
                    int paramPointer = paramInd(layer, layerSize);
                    dim3 dimBlock( 1,1,1 );
                    dim3 dimGrid(layerSize[layer],1,1);
                    forward<<<dimGrid, dimBlock>>>(d_data + inputPointer, layerSize[layer - 1], d_data + outputPointer, layerSize[layer],
                            d_param + paramPointer, Nlayers - layer - 1);
                }

                double *probs;
                probs = (double * ) malloc(layerSize[Nlayers - 1]* sizeof(double));
                int probs_pointer = dataInd(Nlayers - 1, layerSize);
                cudaMemcpy(probs, d_data+probs_pointer, layerSize[Nlayers - 1]*sizeof(double), cudaMemcpyDeviceToHost);

                /* Softmax */
                double sum = 0.0;
                for (int i = 0; i < layerSize[Nlayers - 1]; i++) {
                    sum += exp(probs[i]);
                }
                for (int i = 0; i < layerSize[Nlayers - 1]; i++) {
                    probs[i] = exp(probs[i]) / sum;
                }

                /* Loss computation */
                /* Compute the cross-entropy loss */
                /* Add the log probabilities assigned to the correct classes */
                global_loss += -log(probs[label[batch_index * batch_size + img]]);

                /* Gradient computation for the last (score) layer */
                probs[label[batch_index * batch_size + img]] -= 1.0;

                cudaMemcpy(d_data+probs_pointer, probs, layerSize[Nlayers - 1]*sizeof(double), cudaMemcpyHostToDevice);

                /* Backpropagation */
                for (layer = Nlayers - 1; layer > 0; layer--) {
                    int inputPointer = dataInd(layer, layerSize);
                    int outputPointer = dataInd(layer - 1, layerSize);
                    int paramPointer = paramInd(layer, layerSize);
                    dim3 dimBlock( 1, 1, 1 );
                    dim3 dimGrid( layerSize[layer], 1, 1 );
                    backward_part1<<<dimGrid, dimBlock>>>(d_data + inputPointer, layerSize[layer], d_data + outputPointer, layerSize[layer - 1],
                             d_param + paramPointer, d_grad_param + paramPointer, layer, lambda);
                    if (layer > 1) {
                        dim3 dimBlock( 1, 1, 1 );
                        dim3 dimGrid( layerSize[layer-1], 1, 1 );
                        backward_part2<<<dimGrid, dimBlock>>>(d_data + inputPointer, layerSize[layer], d_data + outputPointer, layerSize[layer - 1],
                             d_param + paramPointer, d_grad_param + paramPointer, layer, lambda);
                    }
                }
            }

            /* Update parameters using gradient averaged over batch)*/
            dim3 dimBlock(1, 1 );
            dim3 dimGrid( num_param, 1 );
            update_param<<<dimGrid, dimBlock>>>(d_param, d_grad_param, num_param, learning_rate, batch_size);

            /* Compute global loss averaged over batch */
            global_loss /= batch_size;
            if (batch_index % 100 == 0) { 
                printf("Iteration - %d, batch - %d, total loss: %f\n", itTotal, batch_index, global_loss); 
            }
        }
    }
    cudaFree(d_data);
    free(images);
}

void test(const char filename[], int* label, double *d_param, int *layerSize, int Nlayers) {

    int it, layer;
    double global_loss = 0.0;
    int image_size = layerSize[0];
    int acc = 0;

    /* Allocate image data array */
    double *images;
    images  = (double *) malloc(image_size* sizeof(double));

    /* Allocate data array */
    double *d_data ;
    cudaMalloc( (void**)&d_data, dataSize(layerSize,Nlayers)*sizeof(double));

    /* Loop over batches */
    for (it = 0; it < TEST_SET_SIZE; it++) {

        /* Load image data */
        read_images(filename, images, image_size, 1, it);

        /* Copy one image to the 1st layer of the data */
        cudaMemcpy(d_data, images, image_size*sizeof(double),cudaMemcpyHostToDevice);

        /* Forward pass */
        for (layer = 1; layer < Nlayers; layer++) {
            int inputPointer = dataInd(layer - 1, layerSize);
            int outputPointer = dataInd(layer, layerSize);
            int paramPointer = paramInd(layer, layerSize);
            dim3 dimBlock( layerSize[layer], 1 );
            dim3 dimGrid( 1, 1 );
            forward<<<dimGrid, dimBlock>>>(d_data + inputPointer, layerSize[layer - 1], d_data + outputPointer, layerSize[layer],
                    d_param + paramPointer, Nlayers - layer - 1);
        }

        /* Softmax */
        double *probs;
        probs = (double * ) malloc(layerSize[Nlayers - 1]* sizeof(double));
        int probs_pointer = dataInd(Nlayers - 1, layerSize);
        cudaMemcpy(probs, d_data+probs_pointer, layerSize[Nlayers - 1]*sizeof(double), cudaMemcpyDeviceToHost);
        
        double sum = 0.0;
        for (int i = 0; i < layerSize[Nlayers-1]; i++) {
            sum += exp(probs[i]);
        }
        double predicted_class_score = -1.0;
        int predicted_class = -1;
        for (int i = 0; i < layerSize[Nlayers-1]; i++) {
            /* Compute normalized probability of the correct class */
            probs[i] = exp(probs[i])/sum;
            /* Argmax */
            if (probs[i] > predicted_class_score) {
                predicted_class_score = probs[i];
                predicted_class = i;
            }
        }

        if (predicted_class == label[it]) {
            acc++;
        }

        /* Compute the cross-entropy loss */
        global_loss += -log(probs[label[it]]);


        if ((it+1) % 1000 == 0) {
            printf("TEST set size: %d\n", it+1);
            printf("TEST total loss: %f\n", global_loss / (it+1));
            printf("TEST accuracy: %f\n", (float)acc / (float)(it+1));
        }

    }
    cudaFree(d_data);
    free(images);
}

int main(int argc, char **argv) {

/* local variables */
    int batch_size, W = 28, H = 28;
    const char train_images_filename[] = "./mnist_data/train_images.csv";
    const char train_labels_filename[] = "./mnist_data/train_labels.csv";
    const char test_images_filename[] = "./mnist_data/test_images.csv";
    const char test_labels_filename[] = "./mnist_data/test_labels.csv";

/* Read batch size from command line */
    if (argc < 2) {
        printf("No batch size given \n");
        exit(1);
    }
    batch_size = atoi(argv[1]);

    printf("batch_Size: %d\n", batch_size);

    /* Set neural network parameters */
    double lambda = 0.0005, learning_rate = 0.01, momentum = 0.9;
    int Nlayers = 4;
    int layerSize[4] = {W*H,300,100,10};
    int pSize = paramSize(layerSize, Nlayers) * sizeof(double);
    double *h_param, *d_param;
    double *h_grad_param, *d_grad_param;
    int *train_label;
    int *test_label;

    /* Allocate memory and read training labels */
    train_label = (int *) malloc(TRAIN_SET_SIZE * sizeof(int));
    read_labels(train_labels_filename, train_label, TRAIN_SET_SIZE);

    /* Allocate memory for parameters */
    h_param = (double *) malloc(pSize);
    h_grad_param = (double *) malloc(pSize);
    cudaMalloc( (void**)&d_param, pSize);
    cudaMalloc( (void**)&d_grad_param, pSize);

    srandom(1);
    /* Initialize parameters using "xavier" initialization */
    for (int layer = 1; layer < Nlayers; layer++) {
        double a = sqrt(3.0/(float)layerSize[layer-1]);
        int param_pointer = paramInd(layer, layerSize);
        for (int i = 0; i < layerSize[layer]; i++) {
            h_param[param_pointer+i*(layerSize[layer-1] +1)] = 0.0; //bias
            for (int j = 1; j < layerSize[layer-1] +1; j++) {
                h_param[param_pointer+i*(layerSize[layer-1] +1)+j] = 2.0*a * (double) random() / RAND_MAX - a;
            }
        }
    }
    /* Initialize gradients with 0 */
    memset(h_grad_param, 0, pSize);

    cudaMemcpy( d_param, h_param, pSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_grad_param, h_grad_param, pSize, cudaMemcpyHostToDevice );

    /* TRAIN */
    train(train_images_filename, train_label, d_param, d_grad_param, layerSize, Nlayers, batch_size, lambda, learning_rate, momentum);

    /* Deallocate arrays */
    free(h_grad_param);
    cudaFree(d_grad_param);
    free(train_label);

    /* Allocate memory and read testing labels */
    test_label = (int *) malloc(TEST_SET_SIZE * sizeof(int));
    read_labels(test_labels_filename, test_label, TEST_SET_SIZE);

    /* TEST */
    test(test_images_filename, test_label, d_param, layerSize, Nlayers);

    /* Deallocate arrays */
    free(h_param);
    cudaFree(d_param);
    free(test_label);

    return 0;
}
