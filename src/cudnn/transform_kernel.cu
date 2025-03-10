
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

  if (!transform->needTransform)
  {
    transform->runtime = 0;
    if (print_cost)
      printf("  measure[Transform]: s(%d %d %d %d) layout(%d -> %d) cost(%.4lf)\n",
        transform->N, transform->C, transform->H, transform->W,
        transform->src_layout, transform->dst_layout, transform->runtime
      );
    return;
  }
  

  // set descriptors
  checkCUDNN(cudnnSetTensor4dDescriptor(inputTensor, transform->src_layout,
    CUDNN_DATA_FLOAT, transform->N, transform->C, transform->H, transform->W));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, transform->dst_layout,
    CUDNN_DATA_FLOAT, transform->N, transform->C, transform->H, transform->W));

  

  checkCUDA(cudaDeviceSynchronize());
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      checkCUDA(cudaEventRecord(startEvent));
    }

    checkCUDNN(cudnnTransformTensor(
      dnn, &alpha, inputTensor, inputPtr,
      &beta, outputTensor, outputPtr));
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
