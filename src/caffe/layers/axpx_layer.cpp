/*
 * Axpxpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

#include "caffe/layers/axpx_layer.hpp"

namespace caffe {

template <typename Dtype>
void AxpxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
  CHECK_EQ(bottom[0]->shape(3), 1);
  CHECK_EQ(bottom[0]->shape(4), 1);
  top[0]->ReshapeLike(*bottom[1]);
  int spatial_dim = bottom[1]->count(3);
  if (spatial_sum_multiplier_.count() < spatial_dim) {
    spatial_sum_multiplier_.Reshape(vector<int>(1, spatial_dim));
    caffe_set(spatial_dim, Dtype(1), 
        spatial_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void AxpxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void AxpxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(AxpxLayer);
#endif

INSTANTIATE_CLASS(AxpxLayer);
REGISTER_LAYER_CLASS(Axpx);

} // namespace
