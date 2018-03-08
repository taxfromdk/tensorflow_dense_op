/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

void DenseKernelLauncher(const float* in, const int N, float* out);

REGISTER_OP("Dense")
  .Input("input: float")
  .Input("weights: float")
  .Input("biases: float")
  .Output("output: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    //printf("register op!!! \n");

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    shape_inference::ShapeHandle biases_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &biases_shape));
    
    

    //shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 1);
    
    shape_inference::DimensionHandle samples = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle units = c->Dim(weight_shape, 1);
    
    c->set_output(0, c->Matrix(samples, units));

    return Status::OK();
  });

class DenseOp : public OpKernel {
public:
  explicit DenseOp(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseOp-Begin\n");

    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
      
    // get the bias tensor
    const Tensor& biases = context->input(2);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    const TensorShape& biases_shape = biases.shape();
    
    //Check that inputs are two dimensional
    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(biases_shape.dims(), 2);
    
    const int batch_samples = input_shape.dim_size(0);
    //printf("batch_samples %d\n", batch_samples);

    const int input_feature_width = input_shape.dim_size(1);
    //printf("input_feature_width %d\n", input_feature_width);

    const int units = weights_shape.dim_size(1);
    //printf("units %d\n", units);

    //Check input width matches weights height 
    DCHECK_EQ(input_feature_width, weights_shape.dim_size(0));
    //Check weights width match bias width 
    DCHECK_EQ(weights_shape.dim_size(1), biases_shape.dim_size(1));

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(units);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto biases_tensor = biases.matrix<float>();
    auto output_tensor = output->matrix<float>();
    
    // Set all but the first element of the output tensor to 0.
    const int N = 1;
    auto f_input = input.flat<float>();
    auto f_output = output->template flat<float>();

    // Call the cuda kernel launcher
    //DenseKernelLauncher(f_input.data(), N, f_output.data());

    
    for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) {
      for (int ix_unit = 0; ix_unit < units; ix_unit++) {
        output_tensor(ix_sample, ix_unit) = 0;
        for (int ix_input = 0; ix_input < input_feature_width; ix_input++) {
          output_tensor(ix_sample, ix_unit) += input_tensor(ix_sample, ix_input) * weights_tensor(ix_input, ix_unit );
        }  
        output_tensor(ix_sample, ix_unit) += biases_tensor(0, ix_unit);
      }
    }
    
  }
};

REGISTER_KERNEL_BUILDER(Name("Dense").Device(DEVICE_CPU), DenseOp);

REGISTER_OP("DenseGrad")
  .Input("grad: float32")
  .Input("input: float32")
  .Input("weights: float32")
  .Input("biases: float32")
  .Output("grad_input: float32")
  .Output("grad_weights: float32")
  .Output("grad_biases: float32");

class DenseGradOp : public OpKernel {
public:
  explicit DenseGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    DCHECK_EQ(4, context->num_inputs());

    const Tensor& grad = context->input(0);
    
    const Tensor& input = context->input(1);
    
    const Tensor& weights = context->input(2);
    
    const Tensor& biases = context->input(3);
    
    TensorShape grad_shape = grad.shape();
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    TensorShape biases_shape = biases.shape();
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    Tensor* grad_biases = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    OP_REQUIRES_OK(context, context->allocate_output(2, biases_shape, &grad_biases));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto input_tensor = input.matrix<float>();
    
    auto grad_input_tensor = grad_input->matrix<float>();
    auto grad_weights_tensor = grad_weights->matrix<float>();
    auto grad_biases_tensor = grad_biases->matrix<float>();

    int input_feature_width = input_shape.dim_size(1);  //Number of values in each sample
    int batch_samples = input_shape.dim_size(0); //Number of samples in batch
    int units = weights_shape.dim_size(1); //Number of units
    
    for (int x = 0; x < units; x++) 
    {
        grad_biases_tensor(0, x) = 0.0;
    }

    for (int x = 0; x < units; x++) //unit index 
    {
        for (int y = 0; y < input_feature_width; y++) //input feature index
        {
            grad_weights_tensor(y, x) = 0.0;
        }
    }
    
    for (int x = 0; x < input_feature_width; x++) 
    {
        for (int y = 0; y < batch_samples; y++) 
        {
            grad_input_tensor(y, x) = 0.0;
        }
    }
    
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
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseGrad").Device(DEVICE_CPU), DenseGradOp);

