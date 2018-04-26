#include <vector>

// #include <cuda.h>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/interp_layer.hpp"

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  int num_specs = 0;
	CHECK_LE(bottom.size(), 2) << "bottom number of interp layer should be 1 or 2 ";
  num_specs += interp_param.has_zoom_factor();
  num_specs += interp_param.has_shrink_factor();
  num_specs += interp_param.has_height() && interp_param.has_width();
	num_specs += (bottom.size() == 2);
  CHECK_EQ(num_specs, 1) << "Output dimension specified either by "
			 << "zoom factor or shrink factor or explicitly";
  pad_beg_ = interp_param.pad_beg();
  pad_end_ = interp_param.pad_end();
  CHECK_LE(pad_beg_, 0) << "Only supports non-pos padding (cropping) for now";
  CHECK_LE(pad_end_, 0) << "Only supports non-pos padding (cropping) for now";
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() == 5) {
    length_ = bottom[0]->shape(2);
    spatial_start_axis_ = 3;
  } else if (bottom[0]->num_axes() == 4) {
    length_ = 1;
    spatial_start_axis_ = 2;
  } else {
    NOT_IMPLEMENTED;
  }
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  height_in_ = bottom[0]->shape(spatial_start_axis_);
  width_in_ = bottom[0]->shape(spatial_start_axis_ + 1); // start here
  height_in_eff_ = height_in_ + pad_beg_ + pad_end_;
  width_in_eff_ = width_in_ + pad_beg_ + pad_end_;
  InterpParameter interp_param = this->layer_param_.interp_param();
  if (interp_param.has_zoom_factor()) {
    const int zoom_factor = interp_param.zoom_factor();
    CHECK_GE(zoom_factor, 1) << "Zoom factor must be positive";
    height_out_ = height_in_eff_ + (height_in_eff_ - 1) * (zoom_factor - 1);
    width_out_ = width_in_eff_ + (width_in_eff_ - 1) * (zoom_factor - 1);
  }
  else if (interp_param.has_shrink_factor()) {
    const int shrink_factor = interp_param.shrink_factor();
    CHECK_GE(shrink_factor, 1) << "Shrink factor must be positive";
    height_out_ = (height_in_eff_ - 1) / shrink_factor + 1;
    width_out_ = (width_in_eff_ - 1) / shrink_factor + 1;
  }
  else if (interp_param.has_height() && interp_param.has_width()) {
    height_out_  = interp_param.height();
    width_out_  = interp_param.width();
  }
	else if (bottom.size() == 2) {
    height_out_  = bottom[1]->shape(spatial_start_axis_);
    width_out_  = bottom[1]->shape(spatial_start_axis_ + 1);
  }
  else {
    LOG(FATAL); // we have already checked for that
  }
  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";
  vector<int> top_shape;
  top_shape.push_back(num_);
  top_shape.push_back(channels_);
  if (bottom[0]->num_axes() == 5)
    top_shape.push_back(length_);
  top_shape.push_back(height_out_);
  top_shape.push_back(width_out_);

  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(InterpLayer);
#endif


INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);

}  // namespace caffe