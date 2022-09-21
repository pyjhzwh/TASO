/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "taso/ops.h"
using namespace taso;


TensorHandle Graph::transform_conv2d(const TensorHandle _input,
                           int _outputC,
                           int _kernelH, int _kernelW,
                           int _strideH, int _strideW,
                           PaddingMode _padding,
                           ActiMode _activation,
                           Layout _dst_layout)
{
  const int dims[4] = {_outputC, _input->dim[1], _kernelH, _kernelW};
  int total = dims[0] * dims[1] * dims[2] * dims[3];
  // Randomly initialize weights
  DATATYPE* data = (DATATYPE*) malloc(total * sizeof(DATATYPE));
  for (int i = 0; i < total; i++)
    data[i] = (DATATYPE)std::rand() / RAND_MAX;
  TensorHandle weight = new_weight(4, dims, data);
  free(data);

  return transform_conv2d(_input, weight, _strideH, _strideW,
                _padding, _activation, _dst_layout);
}


TensorHandle Graph::transform_conv2d(const TensorHandle _input,
                           const TensorHandle _weight,
                           int _strideH, int _strideW,
                           PaddingMode _padding,
                           ActiMode _activation,
                           Layout _dst_layout)
{
  cudnnTensorFormat_t src_layout = getCuDNNLayout((*_input).layout);
  cudnnTensorFormat_t dst_layout = getCuDNNLayout(_dst_layout);
  Op op = model->get_or_create_transform_conv2d(*_input, *_weight, _strideH, _strideW,
                                      _padding, _activation, src_layout, dst_layout);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  add_edge(_weight->op, op, _weight->idx, 1);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

Op Model::get_or_create_transform_conv2d(Tensor _input, Tensor _weight,
                                        int _strideH, int _strideW,
                                        PaddingMode _padding,
                                        ActiMode _activation,
                                        cudnnTensorFormat_t _src_layout,
                                        cudnnTensorFormat_t _dst_layout)
{
  if (_input.dim[1] % _weight.dim[1] != 0)
    return Op::INVALID_OP;
  // key is (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
  //         strideH, strideW, padding, activation, src_layout, dst_layout)
  TransformConv2DKey key(
    _input, _weight, _strideH, _strideW, _padding, _activation, _src_layout, _dst_layout);
  TransformConv2D* convOp;
  if (transform_conv2d.find(key) != transform_conv2d.end()) {
    convOp = transform_conv2d[key];
  } else {
    convOp = new TransformConv2D(this, _input, _weight, _strideH, _strideW,
                        _padding, _activation,_src_layout, _dst_layout);
    measure_transform_conv2d_cost(convOp);
    transform_conv2d[key] = convOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = convOp;
  return ret;
}

TransformConv2D::TransformConv2D(Model* _model, Tensor _input, Tensor _weight,
               int _strideH, int _strideW,
               PaddingMode _padding,
               ActiMode _activation,
               cudnnTensorFormat_t _src_layout,
               cudnnTensorFormat_t _dst_layout)
: Conv2D(_model, _input, _weight, _strideH, _strideW, _padding, _activation)
{
  type = OP_TRANSFORM_CONV2D;
  src_layout =_src_layout;
  dst_layout = _dst_layout;
  if (src_layout == CUDNN_TENSOR_NCHW)
    assert(_input.default_layout());
  else
    assert(_input.is_NHWC_layout());
  if(src_layout == dst_layout)
  {
    needTransform = false;
  }
  else{
    needTransform = true;
  }
}

TransformConv2D::~TransformConv2D(void)
{}

bool TransformConv2D::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_GROUP:
    {
      int inputC = inputs[0].dim[1];
      int weightC = inputs[1].dim[1];
      assert(inputC % weightC == 0);
      *value = inputC / weightC;
      return true;
    }
    case PM_KERNEL_H:
      *value = inputs[1].dim[2];
      return true;
    case PM_KERNEL_W:
      *value = inputs[1].dim[3];
      return true;
    case PM_STRIDE_H:
      *value = strideH;
      return true;
    case PM_STRIDE_W:
      *value = strideW;
      return true;
    case PM_PAD:
      *value = padding;
      return true;
    case PM_ACTI:
      *value = (int) activation;
      return true;
    // Layout
    case PM_SRCLAYOUT:
      *value = src_layout;
      return true;
    case PM_DSTLAYOUT:
      *value = dst_layout;
      return true;
    default:
      return OpBase::get_int_parameter(para, value);
  }
}

void TransformConv2D::collect_costs(float& exe_time, float& flops,
                           float& mem_acc, int& num_kernels)
{
  size_t outputSize = outputs[0].volume() * sizeof(DATATYPE);
  size_t inputSize = inputs[0].volume() * sizeof(DATATYPE);
  size_t weightSize = inputs[1].volume() * sizeof(DATATYPE);
  // cost metrics
  exe_time += runtime;
  int kernelH = inputs[1].dim[2];
  int kernelW = inputs[1].dim[3];
  int inputC = inputs[1].dim[1];
  flops += outputSize * (kernelH * kernelW * inputC + 1);
  if (activation != AC_MODE_NONE)
    flops += outputSize;
  mem_acc += inputSize + outputSize + weightSize;
  num_kernels += 1;
  printf("        cost[TransformConv2D]: i(%d %d %d %d) w(%d %d %d %d) s(%d %d) p(%d) cost(%.4lf) total_cost(%.4lf)\n",
          inputs[0].dim[0], inputs[0].dim[1], inputs[0].dim[2], inputs[0].dim[3],
          inputs[1].dim[0], inputs[1].dim[1], inputs[1].dim[2], inputs[1].dim[3],
          strideH, strideW, padding, runtime, exe_time);
}

// keys are (inputN, inputC, inputH, inputW, outputC, kernelH, kernelW,
//           strideH, strideW, padding, acitvation,
//           input.split[0], weight.split[0])
TransformConv2DKey::TransformConv2DKey(Tensor _input, Tensor _weight,
                     int _strideH, int _strideW,
                     PaddingMode _padding,
                     ActiMode _activation,
                     cudnnTensorFormat_t _src_layout,
                     cudnnTensorFormat_t _dst_layout
                     )
{
  assert(_input.dim[1] % _weight.dim[1] == 0);
  int groups = _input.dim[1] / _weight.dim[1];
  assert(_weight.dim[0] % groups == 0);
  int idx = 0;
  keys[idx++] = _strideH;
  keys[idx++] = _strideW;
  keys[idx++] = _padding;
  keys[idx++] = _activation;
  keys[idx++] = _src_layout;
  keys[idx++] = _dst_layout;
  _input.serialize(keys, idx);
  _weight.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(KEY_LENGTH == idx);
}

