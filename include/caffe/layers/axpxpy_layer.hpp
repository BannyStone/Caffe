/*
 * Axpy Layer
 *
 * Created on: May 1, 2017
 * Author: hujie
 */

#ifndef CAFFE_AXPXPY_LAYER_HPP_
#define CAFFE_AXPXPY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

/**
 * @brief For reduce memory and time both on training and testing, we combine
 *        channel-wise scale operation and element-wise addition operation 
 *        into a single layer called "axpxpy".
 *       
 */
template <typename Dtype>
class AxpxpyLayer: public Layer<Dtype> {
 public:
  explicit AxpxpyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Axpxpy"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
/**
 * @param Formulation:
 *            F = a * X + X + Y
 *	  Shape info:
 *            a:  N x C x T x 1 x 1  --> bottom[0]      
 *            X:  N x C x T x H x W  --> bottom[1]       
 *            Y:  N x C x T x H x W  --> bottom[2]     
 *            F:  N x C x T x H x W  --> top[0]
 */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> spatial_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_AXPXPY_LAYER_HPP_