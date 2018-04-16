#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/corrv1_layer.hpp"

namespace caffe {

template <typename Dtype>
void Corrv1Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  Corrv1Parameter corr_param = this->layer_param_.corrv1_param();
  
  CHECK(corr_param.has_kernel_size()) << "Filter kernel_size is not set";
  CHECK(corr_param.has_max_displacement()) << "Max displacement is required.";

  template_type_ = corr_param.template_type();
  template_index_ = -1;
  if (template_type_ == Corrv1Parameter_TemplateType_INDEX)
    template_index_ = corr_param.template_index();

  kernel_size_ = corr_param.kernel_size();
  if(kernel_size_ % 2 == 0) LOG(FATAL) << "Odd kernel size required";
  
  max_displacement_ = corr_param.max_displacement();
  pad_size_ = corr_param.pad();
  stride1_ = corr_param.stride_1();
  stride2_ = corr_param.stride_2();
  
  do_abs_ = corr_param.do_abs();
  
  corr_type_ = corr_param.correlation_type();
  
  LOG(INFO) << "Kernel Size: " << kernel_size_;
  LOG(INFO) << "Stride 1: " << stride1_;
  LOG(INFO) << "Stride 2: " << stride2_;
  LOG(INFO) << "Max Displacement: " << max_displacement_;
  
}

template <typename Dtype>
void Corrv1Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->shape(0);
  
  int bottomchannels = bottom[0]->shape(1);
  int bottomlength = bottom[0]->shape(2);
  
  int paddedbottomheight = bottom[0]->shape(3)+2*pad_size_;
  int paddedbottomwidth = bottom[0]->shape(4)+2*pad_size_;
  
  // Size computation
  kernel_radius_ = (kernel_size_ - 1) / 2; //size of unreachable border region (on each side)
  border_size_ = max_displacement_ + kernel_radius_; //size of unreachable border region (on each side)
  
  top_length_ = bottomlength;
  top_width_ = ceil((float)(paddedbottomwidth - border_size_*2) / (float)stride1_);
  top_height_ = ceil((float)(paddedbottomheight - border_size_*2) / (float)stride1_);
  
  CHECK_GE(top_length_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  CHECK_GE(top_width_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  CHECK_GE(top_height_, 1) << "Correlation cannot be done with current settings. Neighborhood and kernel don't fit in blob";
  
  // Given a center position in image 1, how many displaced positions in -x / +x direction do we consider in image 2 (neighborhoodGridWidth):
  neighborhood_grid_radius_ = max_displacement_ / stride2_;
  neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
  
  // Top Channels amount to displacement combinations in X and Y direction:
  top_channels_ = neighborhood_grid_width_ * neighborhood_grid_width_;
  
  //Reshape top
  vector<int> shape_top;
  shape_top.push_back(num_);
  shape_top.push_back(top_channels_);
  shape_top.push_back(top_length_);
  shape_top.push_back(top_height_);
  shape_top.push_back(top_width_);
  top[0]->Reshape(shape_top);

  // rbots (These are the blobs that store the padded and dimension rearranged data
  template_.reset(new Blob<Dtype>());
  template_->Reshape(num_, paddedbottomheight, paddedbottomwidth, bottomchannels);
  rbot_.reset(new Blob<Dtype>());
  vector<int> shape_rearr;
  shape_rearr.push_back(num_);
  shape_rearr.push_back(bottomlength); 
  shape_rearr.push_back(paddedbottomheight);
  shape_rearr.push_back(paddedbottomwidth);
  shape_rearr.push_back(bottomchannels);
  rbot_->Reshape(shape_rearr);
  
  rtopdiff_.reset(new Blob<Dtype>());
  vector<int> shape_rearr_top;
  shape_rearr_top.push_back(num_);
  shape_rearr_top.push_back(top_length_);
  shape_rearr_top.push_back(top_height_);
  shape_rearr_top.push_back(top_width_);
  shape_rearr_top.push_back(top_channels_);
  rtopdiff_->Reshape(shape_rearr_top);
}

template <typename Dtype>
void Corrv1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Corrv1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(Corrv1Layer);
#endif

INSTANTIATE_CLASS(Corrv1Layer);
REGISTER_LAYER_CLASS(Corrv1);

}  // namespace caffe
