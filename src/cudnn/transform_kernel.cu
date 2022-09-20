
#include "taso/ops.h"
#include "taso/cuda_helper.h"
using namespace taso;

void Transform::map(void)
{
  if (src_layout == dst_layout)
    return;
  //TODO: for now the output and input share the same instance
  checkCUDNN(cudnnCreateTensorDescriptor(&srcTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&dstTensor));

  if (src_layout == CUDNN_TENSOR_NCHW)
  {
    N = inputs[0].dim[0];
    C = inputs[0].dim[1];
    H = inputs[0].dim[2];
    W = inputs[0].dim[3];
  }
  else if (src_layout == CUDNN_TENSOR_NHWC)
  {
    N = inputs[0].dim[0];
    H = inputs[0].dim[1];
    W = inputs[0].dim[2];
    C = inputs[0].dim[3];
  }
  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(srcTensor, src_layout,
    CUDNN_DATA_FLOAT, N, C, H, W));
  checkCUDNN(cudnnSetTensor4dDescriptor(dstTensor, dst_layout,
    CUDNN_DATA_FLOAT, N, C, H, W));

  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * N * C * H * W;
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}  

void Transform::unmap(void)
{
  checkCUDNN(cudnnDestroyTensorDescriptor(srcTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(dstTensor));
  // free tensors
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

void Transform::forward(bool block)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkCUDNN(cudnnTransformTensor(
    model->dnn, &alpha, srcTensor, inputs[0].data_ptr,
    &beta, dstTensor, outputs[0].data_ptr));

  if (block)
    checkCUDA(cudaDeviceSynchronize());
}

void Model::measure_transform_cost(Transform* transform)
{
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(transform->srcTensor, transform->src_layout,
    CUDNN_DATA_FLOAT, transform->N, transform->C, transform->H, transform->W));
  checkCUDNN(cudnnSetTensor4dDescriptor(transform->dstTensor, transform->dst_layout,
    CUDNN_DATA_FLOAT, transform->N, transform->C, transform->H, transform->W));

  

  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      checkCUDA(cudaEventRecord(startEvent));
    }

    checkCUDNN(cudnnTransformTensor(
      dnn, &alpha, transform->srcTensor, transform->inputs[0].data_ptr,
      &beta, transform->dstTensor, transform->outputs[0].data_ptr));
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  transform->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Transform]: s(%d %d %d %d) layout(%d -> %d) cost(%.4lf)\n",
      transform->N, transform->C, transform->H, transform->W,
      transform->src_layout, transform->dst_layout, transform->runtime
    );
}
