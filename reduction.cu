//basic cpp includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>
//Last one to accumulate the results via accumulate function
#include <numeric>
//cuda includes
#include "cuda_runtime.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cassert>


//I decided this time to use another algorithm and the host for the single threaded calculation as I find it more reliable for comparing

int singleThreadedSum (float tab[], int len)
{
int res = 0;
        for (int i = 0 ; i < len; i++)
        {
                res += tab[i];
        }
return res;
}



__global__ static void reductionSum(const float *input, float *output, int lengthTab)
{
    extern __shared__ float partSum[];

    unsigned int th = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int index;

    //Handle of outbound values (0)
    partSum[th] = (i < lengthTab ? input[i] : 0);
    partSum[th + blockDim.x] = (i + blockDim.x*gridDim.x < lengthTab ? input[i + blockDim.x* gridDim.x] : 0);
    __syncthreads();
    //Reduction loop using the nvidia algorithm, I couldn't find a way to use: th%stride==0, so I inspired myself on Nvidia documentation on reduction
    // link : https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf page 11
    for (int stride = 1; stride < blockDim.x*2; stride *= 2)
    {
        //Sync at the beginning and not the end!
        __syncthreads();
        index = 2*stride*th;
        if (index < blockDim.x*2)
        {
            partSum[index] += partSum[index + stride];
        }
    }
    __syncthreads();
    if (th == 0){ //Committing to vram to retrieve the results of each block on the host
        output[blockIdx.x] = partSum[0];
    }
}
void wrapper(float *input, int lengthTab, int NUM_THREADS, int NUM_BLOCKS)
{
    float *dinput = NULL;
    float *doutput = NULL;
    float *output = NULL;
    //Allocating memory for the output data
    output = (float *)malloc(NUM_BLOCKS * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&dinput, sizeof(float) * lengthTab));
    checkCudaErrors(cudaMalloc((void **)&doutput, sizeof(float) * NUM_BLOCKS));
    checkCudaErrors(cudaMemcpy(dinput, input, sizeof(float) * lengthTab , cudaMemcpyHostToDevice));
     //Allocating CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
    //Record start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    reductionSum<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS >>>(dinput, doutput, lengthTab);
    cudaDeviceSynchronize();

    //Record stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    //Wait for the stp event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaMemcpy(output, doutput, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost));
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float totalRes = std::accumulate(output, output + NUM_BLOCKS, int(0));
    printf("The result of the multithreaded function is: %f \n", totalRes);
    printf("Elapsed Time for reduction function to complete is : %f msec \n", msecTotal);

    checkCudaErrors(cudaFree(dinput));
    checkCudaErrors(cudaFree(doutput));


    int singleThreadRes = singleThreadedSum(input, lengthTab);

    printf("The result on the single thread function is: %d, Time : %d \n", singleThreadRes, msecTotal);
    free(output);
}

int main(int argc, char **argv)
{
    int lengthTab;
    if (checkCmdLineFlag(argc, (const char **)argv, "nb")) {
        lengthTab = getCmdLineArgumentInt(argc, (const char **)argv, "nb");
    }
    //Numthreads needs to be a power of 2 multiple of 32 and max 1024
    int NUM_THREADS =  256;
     //Multiple of the sm number (30)
    int NUM_BLOCKS = min((lengthTab/NUM_THREADS), 30);
//atleast one block
if (NUM_BLOCKS == 0){ NUM_BLOCKS = 1;}
int dev = findCudaDevice(argc, (const char **)argv);
float *input = NULL;
//Allocating memory for the input data
input = (float *)malloc(lengthTab * sizeof(float));

//Generating the values of the array
for (int i = 0; i < lengthTab; i++)
{
    input[i] = rand() % 10;
}
wrapper(input, lengthTab, NUM_THREADS, NUM_BLOCKS);
free(input);
return EXIT_SUCCESS;
}
