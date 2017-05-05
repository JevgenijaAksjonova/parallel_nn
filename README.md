# A study of parallel feed forward neural networks

In this project, parallelization is used to boost the time performance of a fully-connected feed forward neural network solving a handwritten digit recognition problem using the MNIST dataset. Two implementations of the algorithm are presented: a MPI implementation, which is executed on several CPU cores across multiple processors, and a GPU-based implementation, using the CUDA C library.

### Data

The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is used to test the network implementations. The unzipped data must have the following path: "./mnist_data/train_images.csv"

### Compiling and running instructions

MPI version
```
module add i-compilers intelmpi
mpiicc -o mpi mpi.c -O3
mpirun -np <num_proc> ./mpi <num_epochs>
```

CUDA version
```
module add cuda
nvcc -arch=sm_37 cuda.cu -o cuda.x
./cuda.x <num_epochs>
```

