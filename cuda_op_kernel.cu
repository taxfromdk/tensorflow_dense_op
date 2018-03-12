#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 512

__global__ void DenseKernel(
    const double* inputs,
    const double* weights,
    const double* biases,
    const int batch_samples, 
    const int units, 
    const int input_feature_width, 
    double* output) 
{ 
    //for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) 
    //{
    //    for (int ix_unit = 0; ix_unit < units; ix_unit++) 
    //    {
    //        output_tensor(ix_sample, ix_unit) = 0;
    //        for (int ix_input = 0; ix_input < input_feature_width; ix_input++)
    //        {
    //            output_tensor(ix_sample, ix_unit) += input_tensor(ix_sample, ix_input) * weights_tensor(ix_input, ix_unit );
    //        }  
    //        output_tensor(ix_sample, ix_unit) += biases_tensor(0, ix_unit);
    //    }
    //}

    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch_samples)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = 0.0;
        for (int ix_input = 0; ix_input < input_feature_width; ix_input++)
        {
            output[ix] += inputs[ix_sample*input_feature_width+ix_input] * weights[ix_input*units+ix_unit];
        }  
        output[ix] += biases[ix_unit];          
    }
}

void DenseKernelLauncher(
        const double* inputs, 
        const double* weights,
        const double* biases,
        const int batch_samples, 
        const int units, 
        const int input_feature_width,
        double* output) 
{
    DenseKernel<<<batch_samples,units>>>(inputs, weights, biases, batch_samples, units, input_feature_width, output);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}


/*
for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) {
  for (int ix_unit = 0; ix_unit < units; ix_unit++) {
    //output_tensor(ix_sample, ix_unit) = 0;
    for (int ix_input = 0; ix_input < input_feature_width; ix_input++) {
        //!!!output_tensor(ix_sample, ix_unit) += input_tensor(ix_sample, ix_input) * weights_tensor(ix_input, ix_unit );
        grad_input_tensor(ix_sample, ix_input) += weights_tensor(ix_input, ix_unit )*grad_tensor(ix_sample, ix_unit);
        grad_weights_tensor(ix_input, ix_unit ) += input_tensor(ix_sample, ix_input)*grad_tensor(ix_sample, ix_unit);
    }  
    //!!!output_tensor(ix_sample, ix_unit) += biases_tensor(0, ix_unit);
    grad_biases_tensor(0, ix_unit) += grad_tensor(ix_sample, ix_unit);
  }
}
*/

// Input gradient

__global__ void InputKernel(
    const double* grads,
    const double* weights,
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_inputs) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch_samples*input_feature_width)
    {
        int ix_input = ix % input_feature_width;
        int ix_sample = ix / input_feature_width ;
        grad_inputs[ix] = 0.0;
        
        //sample //unit //input

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {
            grad_inputs[ix_sample*input_feature_width+ix_input] += weights[ix_input*units+ ix_unit]*grads[ix_sample*units+ix_unit];
        }  
    }
}

void InputGradKernelLauncher(
    const double* grads, 
    const double* weights, 
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_inputs)
{
    InputKernel<<<batch_samples,input_feature_width>>>(grads, weights, input_feature_width, batch_samples, units, grad_inputs);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}

// Weights gradient

__global__ void WeightsKernel(
    const double* grads,
    const double* inputs,
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_weights) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_feature_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = 0.0;

        //sample //unit //input

        for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++)
        {
            grad_weights[ix] += inputs[input_feature_width*ix_sample+ix_input]*grads[ix_sample*units+ix_unit];
        }  
    }
}

void WeightsGradKernelLauncher(
    const double* grads, 
    const double* inputs, 
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_weights)
{
    WeightsKernel<<<units,input_feature_width>>>(grads, inputs, input_feature_width, batch_samples, units, grad_weights);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}

// Bias gradient

__global__ void BiasesKernel(
    const double* grads,
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_biases) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units)
    {
        int ix_unit = ix;
        grad_biases[ix_unit] = 0.0;
        for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++)
        {
            grad_biases[ix] += grads[ix_sample*units+ix_unit];
        }  
    }
}

void BiasesGradKernelLauncher(
    const double* grads, 
    const int input_feature_width, 
    const int batch_samples, 
    const int units, 
    double* grad_biases)
{
    BiasesKernel<<<1,units>>>(grads, input_feature_width, batch_samples, units, grad_biases);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
    {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}
