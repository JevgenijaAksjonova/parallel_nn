/**
 * CUDA implementation of a fully-connected feed forward neural network
 *
 * Authors: Jevgenija Aksjonova (jevaks@kth.se)
 *          Beatrice Ionascu (bionascu@kth.se)
 *
 * Last changed: 04/30/2017
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* Compute local number of parameters (includes weights and biases) */
int param_size(int *layer_size, int num_layers) {
    int size = 0;
    size_t i;
    for (i = 1; i < num_layers; i++) {
        size = size + layer_size[i] * (layer_size[i-1] +1);
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
int param_ind(int layer, int *layer_size) {
    int ind = 0;
    size_t i;
    for (i = 1; i < layer; i++) {
        ind += layer_size[i] * (layer_size[i - 1] + 1);
    }
    return ind;
}


/* Compute local starting index of the data corresponding to layer */
int data_ind(int layer, int *layer_size) {
    int ind = 0;
    size_t i;
    for (i = 0; i < layer; i++) {
        ind += layer_size[i];
    }
    return ind;
}


/* Forward pass */
__global__
void forward(double *input, int inputSize, double *output, int outputSize, double *param, int fun) {

    size_t i, j;
    double l_output;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < outputSize ) {

        /* Initialize output with the bias term */
        l_output = param[i];
        for (j = 0; j < inputSize; j++) {
            // Add weighted inputs
            l_output += param[(j+1)*outputSize + i] * input[j];
        }
        /* Activation */
        if (fun > 0) {
            if (FUNCTION_TYPE == 0) {
                l_output = tanh(l_output);
            }
            else if (FUNCTION_TYPE == 1) { // relu function
                if (l_output <= 0) {
                    l_output = 0.0;
                }
            }
        }
        output[i] = l_output;
    }
}


/* Backward pass (backpropagate gradients from current_layer to prev_layer) */
__global__
void backward_part1(double *current_layer, int layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {
    
    size_t i, j;
    double l_current_layer;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < layer_size) {
        l_current_layer = current_layer[i];
        // Update parameters
        grad_param[i] += l_current_layer;
        for (j = 0; j < prev_layer_size; j++) {
            grad_param[(j+1) * layer_size + i ] += l_current_layer * prev_layer[j] + 
                        2.0 * lambda * param[(j+1) * layer_size + i]; // regularization
        }
    }
}

__global__
void backward_part2(double *current_layer, int layer_size, double *prev_layer, int prev_layer_size,
              double *param, double *grad_param, int layer,double lambda) {

    size_t i, k;
    double l_prev_layer;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < prev_layer_size ) {
        l_prev_layer = prev_layer[i];
        // backprapogate partial derivatives
        if (FUNCTION_TYPE == 0) {
            double localGrad = (1-pow(l_prev_layer,2)); // tanh derivative
            l_prev_layer = 0.0; 
            for (k = 0; k < layer_size; k++) {
                l_prev_layer += current_layer[k] * param[(i+1) * layer_size + k];
            }
            l_prev_layer *= localGrad;
        } else if (FUNCTION_TYPE == 1) {
            if (l_prev_layer >0.0) {
                l_prev_layer = 0.0;
                for (k = 0; k < layer_size; k++) {
                    l_prev_layer += current_layer[k] * param[(i+1) * layer_size + k];
                }
            } else {
                l_prev_layer = 0.0;
            }
        }
        prev_layer[i] = l_prev_layer;
    }
}

__global__
void update_param(double *param, double *grad_param, int num_param, double learning_rate, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_param) {
        param[i] = param[i] - learning_rate * grad_param[i] / (float)batch_size;
    }
}

/* Train the network */
void train(const char filename[], int* label, double *d_param, double *d_grad_param, int *layer_size, int num_layers,
           int epochs, double lambda, double learning_rate) {

    size_t img, ep = 0;
    int i, layer, batch_index;
    int input_pointer, output_pointer, param_pointer;
    double sum, global_loss = 1;
    int image_size = layer_size[0];
    int num_param = param_size(layer_size, num_layers);

    /* Allocate image data array */
    double *images;
    images = (double *) malloc(BATCH_SIZE * image_size * sizeof(double));

    /* Allocate data array */
    double *d_data ;
    cudaMalloc( (void**)&d_data, data_size(layer_size, num_layers) * sizeof(double));

    /* Loop over epochs (one epoch = loop over all images) */
    while (ep++ < epochs && global_loss > TOL) {
        /* Loop over batches */
        for (batch_index = 0; batch_index < TRAIN_SET_SIZE / BATCH_SIZE; batch_index++) {
            /* Initialize loss */
            global_loss = 0.0;

            /* Reset gradients to 0 */
            cudaMemset(d_grad_param, 0.0, num_param * sizeof(double));

            /* Load images in batch */
            read_images(filename, images, image_size, BATCH_SIZE, batch_index);

            /* Loop over images in batch */
            for (img = 0; img < BATCH_SIZE; img++) {

                /* Copy one image to the 1st layer of the data */
                cudaMemcpy(d_data, images + img * image_size, image_size * sizeof(double), cudaMemcpyHostToDevice );

                /* Forward pass */
                for (layer = 1; layer < num_layers; layer++) {
                    input_pointer = data_ind(layer - 1, layer_size);
                    output_pointer = data_ind(layer, layer_size);
                    param_pointer = param_ind(layer, layer_size);
                    dim3 dimBlock( 128,1,1 );
                    dim3 dimGrid(ceil((float)layer_size[layer]/128.0),1,1);
                    forward<<<dimGrid, dimBlock>>>(d_data + input_pointer, layer_size[layer - 1], d_data + output_pointer, layer_size[layer],
                            d_param + param_pointer, num_layers - layer - 1);
                }

                double *probs;
                probs = (double * ) malloc(layer_size[num_layers - 1]* sizeof(double));
                int probs_pointer = data_ind(num_layers - 1, layer_size);
                cudaMemcpy(probs, d_data + probs_pointer, layer_size[num_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);

                /* Softmax */
                sum = 0.0;
                for ( i = 0; i < layer_size[num_layers - 1]; i++) {
                    sum += exp(probs[i]);
                }
                for ( i = 0; i < layer_size[num_layers - 1]; i++) {
                    probs[i] = exp(probs[i]) / sum;
                }

                /* Loss computation */
                /* Compute the cross-entropy loss by adding the log probabilities assigned to the correct classes */
                global_loss += -log(probs[label[batch_index * BATCH_SIZE + img]]);

                /* Gradient computation for the last (score) layer */
                probs[label[batch_index * BATCH_SIZE + img]] -= 1.0;

                cudaMemcpy(d_data+probs_pointer, probs, layer_size[num_layers - 1]*sizeof(double), cudaMemcpyHostToDevice);

                /* Backpropagation */
                for (layer = num_layers - 1; layer > 0; layer--) {
                    input_pointer = data_ind(layer, layer_size);
                    output_pointer = data_ind(layer - 1, layer_size);
                    param_pointer = param_ind(layer, layer_size);
                    dim3 dimBlock( 128, 1, 1 );
                    dim3 dimGrid( ceil((float)layer_size[layer]/128.0), 1, 1 );
                    backward_part1<<<dimGrid, dimBlock>>>(d_data + input_pointer, layer_size[layer], d_data + output_pointer, layer_size[layer - 1],
                             d_param + param_pointer, d_grad_param + param_pointer, layer, lambda);
                    if (layer > 1) {
                        dim3 dimBlock( 128, 1, 1 );
                        dim3 dimGrid( ceil((float)layer_size[layer - 1]/128.0), 1, 1 );
                        backward_part2<<<dimGrid, dimBlock>>>(d_data + input_pointer, layer_size[layer], d_data + output_pointer, layer_size[layer - 1],
                             d_param + param_pointer, d_grad_param + param_pointer, layer, lambda);
                    }
                }
            }

            /* Update parameters using gradient averaged over batch)*/
            dim3 dimBlock(512, 1, 1 );
            dim3 dimGrid( ceil((float)num_param/512.0), 1 ,1 );
            update_param<<<dimGrid, dimBlock>>>(d_param, d_grad_param, num_param, learning_rate, BATCH_SIZE);

            /* Compute global loss averaged over batch */
            global_loss /= BATCH_SIZE;
            if (batch_index % 100 == 0) {
                printf("Ep: %zu/%d\t batch:%d \t train loss: %f\n", ep, epochs, batch_index, global_loss);
            }
        }
    }
    cudaFree(d_data);
    free(images);
}


/* Test the network */
void test(const char filename[], int* label, double *d_param, int *layer_size, int num_layers) {
    int it, layer, i;
    int input_pointer, output_pointer, param_pointer, probs_pointer, predicted_class;
    double sum, predicted_class_score, global_loss = 0.0;
    int image_size = layer_size[0];
    int acc = 0;

    /* Allocate image data array */
    double *images;
    images  = (double *) malloc(image_size* sizeof(double));

    /* Allocate data array */
    double *d_data ;
    cudaMalloc( (void**)&d_data, data_size(layer_size,num_layers)*sizeof(double));

    /* Loop over batches */
    for (it = 0; it < TEST_SET_SIZE; it++) {

        /* Load image data */
        read_images(filename, images, image_size, 1, it);

        /* Copy one image to the 1st layer of the data */
        cudaMemcpy(d_data, images, image_size * sizeof(double),cudaMemcpyHostToDevice);

        /* Forward pass */
        for (layer = 1; layer < num_layers; layer++) {
            input_pointer = data_ind(layer - 1, layer_size);
            output_pointer = data_ind(layer, layer_size);
            param_pointer = param_ind(layer, layer_size);
            dim3 dimBlock( 128, 1, 1 );
            dim3 dimGrid( ceil((float)layer_size[layer]/128.0), 1, 1 );
            forward<<<dimGrid, dimBlock>>>(d_data + input_pointer, layer_size[layer - 1], d_data + output_pointer, layer_size[layer],
                    d_param + param_pointer, num_layers - layer - 1);
        }

        /* Softmax and Predict class */
        double *probs;
        probs = (double * ) malloc(layer_size[num_layers - 1] * sizeof(double));
        probs_pointer = data_ind(num_layers - 1, layer_size);
        cudaMemcpy(probs, d_data+probs_pointer, layer_size[num_layers - 1]*sizeof(double), cudaMemcpyDeviceToHost);
        
        sum = 0.0;
        for (i = 0; i < layer_size[num_layers - 1]; i++) {
            sum += exp(probs[i]);
        }
        predicted_class_score = -1.0;
        predicted_class = -1;
        for (i = 0; i < layer_size[num_layers - 1]; i++) {
            /* Compute normalized probability of the correct class */
            probs[i] = exp(probs[i])/sum;
            /* Argmax */
            if (probs[i] > predicted_class_score) {
                predicted_class_score = probs[i];
                predicted_class = i;
            }
        }

        /* Compute accuracy */
        if (predicted_class == label[it]) {
            acc++;
        }

        /* Compute the cross-entropy loss by adding the log probabilities assigned to the correct classes */
        global_loss += -log(probs[label[it]]);

        if ((it+1) % 1000 == 0) {
            printf("Test set size: %d\tloss: %f\taccuracy:%f\n",
                   it + 1, global_loss / (it + 1), (float) acc / (float) (it + 1));
        }

    }
    cudaFree(d_data);
    free(images);
}



int main(int argc, char **argv) {

    /* local variables */
    int epochs, W = 28, H = 28;
    const char train_images_filename[] = "./mnist_data/train_images.csv";
    const char train_labels_filename[] = "./mnist_data/train_labels.csv";
    const char test_images_filename[] = "./mnist_data/test_images.csv";
    const char test_labels_filename[] = "./mnist_data/test_labels.csv";

    /* Read number of epochs from command line */
    if (argc < 2) {
        printf("Number of epochs is not given \n");
        exit(1);
    }
    epochs = atoi(argv[1]);

    /* Set neural network parameters */
    double lambda = 0.0005, learning_rate = 0.01;
    int num_layers = 4;
    int layer_size[4] = {W*H,300,100,10};
    int p_size = param_size(layer_size, num_layers) * sizeof(double);
    double *h_param, *d_param;
    double *h_grad_param, *d_grad_param;
    int *train_label, *test_label;

    /* Allocate memory and read training labels */
    train_label = (int *) malloc(TRAIN_SET_SIZE * sizeof(int));
    read_labels(train_labels_filename, train_label, TRAIN_SET_SIZE);

    /* Allocate memory for parameters */
    h_param = (double *) malloc(p_size);
    h_grad_param = (double *) malloc(p_size);
    cudaMalloc( (void**)&d_param, p_size);
    cudaMalloc( (void**)&d_grad_param, p_size);

    srandom(1);
    /* Initialize parameters using "xavier" initialization */
    int layer, i, j;
    for (layer = 1; layer < num_layers; layer++) {
        double a = sqrt(3.0/(float)layer_size[layer-1]); // uniform interval limit
        int param_pointer = param_ind(layer, layer_size);
        for (i = 0; i < layer_size[layer]; i++) {
            h_param[param_pointer + i] = 0.0; // bias
        }
        for (j = 1; j < layer_size[layer - 1] +1; j++) {
            for (i = 0; i < layer_size[layer]; i++) {
                h_param[param_pointer + j * layer_size[layer] + i] = 2.0 * a * (double) random() / RAND_MAX - a;
            }
        }
    }

    /* Initialize gradients with 0 */
    memset(h_grad_param, 0, p_size);

    cudaMemcpy( d_param, h_param, p_size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_grad_param, h_grad_param, p_size, cudaMemcpyHostToDevice );

    /* Train network */
    train(train_images_filename, train_label, d_param, d_grad_param, layer_size, num_layers, epochs, lambda, learning_rate);

    /* Deallocate arrays */
    free(h_grad_param);
    cudaFree(d_grad_param);
    free(train_label);

    /* Allocate memory and read testing labels */
    test_label = (int *) malloc(TEST_SET_SIZE * sizeof(int));
    read_labels(test_labels_filename, test_label, TEST_SET_SIZE);

    /* Test network */
    test(test_images_filename, test_label, d_param, layer_size, num_layers);

    /* Deallocate arrays */
    free(h_param);
    cudaFree(d_param);
    free(test_label);

    return 0;
}
