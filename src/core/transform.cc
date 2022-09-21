

#include "taso/ops.h"
using namespace taso;


TensorHandle Graph::transform(const TensorHandle _input,
                              const Layout _src_layout,
                              const Layout _dst_layout)
{
  cudnnTensorFormat_t src_layout = getCuDNNLayout(_src_layout);
  cudnnTensorFormat_t dst_layout = getCuDNNLayout(_dst_layout);
  Op op = model->get_or_create_transform(*_input, src_layout, dst_layout);
  assert(op != Op::INVALID_OP);
  add_edge(_input->op, op, _input->idx, 0);
  TensorHandle t = new Tensor(op.ptr->outputs[0]);
  t->op = op;
  return t;
}

cudnnTensorFormat_t Graph::getCuDNNLayout(Layout layout)
{
  if (layout == LAYOUT_NCHW)
    return CUDNN_TENSOR_NCHW;
  else if (layout == LAYOUT_NHWC)
    return CUDNN_TENSOR_NHWC;
  else {
    assert(false);
  }
}


Op Model::get_or_create_transform(Tensor _input, cudnnTensorFormat_t _src_layout,
                                  cudnnTensorFormat_t _dst_layout)
{
  TransformKey key(_input, _src_layout, _dst_layout);
  Transform* transformOp;
  if (transform.find(key) != transform.end()) {
    transformOp = transform[key];
  } else {
    transformOp = new Transform(this, _input, _src_layout, _dst_layout);
    measure_transform_cost(transformOp);
    transform[key] = transformOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = transformOp;
  return ret;
}

Transform::Transform(Model* _model, Tensor _input,
                     cudnnTensorFormat_t _src_layout,
                     cudnnTensorFormat_t _dst_layout)
: OpBase(_input, _model, OP_TRANSFORM),
    src_layout(_src_layout), dst_layout(_dst_layout)
{
  
  needTransform = (src_layout != dst_layout);
  numOutputs = 1;
  // set dims and strides
  outputs[0].numDim = _input.numDim;
  for (int i = 0; i <  _input.numDim; i++) {
    outputs[0].dim[i] = _input.dim[i];
    outputs[0].stride[i] = _input.stride[i];
    outputs[0].split[i] = _input.split[i];
  }
  outputs[0].idx = 0;
  N = inputs[0].dim[0];
  C = inputs[0].dim[1];
  H = inputs[0].dim[2];
  W = inputs[0].dim[3];
}

Transform::~Transform(void)
{}

bool Transform::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
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

void Transform::collect_costs(float& exe_time, float& flops,
                              float& mem_acc, int& num_kernels)
{
    exe_time += runtime;
}

TransformKey::TransformKey(Tensor _input,
                           cudnnTensorFormat_t _src_layout,
                           cudnnTensorFormat_t _dst_layout)
{
  int idx = 0;
  keys[idx++] = _src_layout;
  keys[idx++] = _dst_layout;
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

