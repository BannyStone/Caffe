#include <algorithm>
#include <vector>

#include "caffe/common.cuh"
// #include <cuda.h>

#include "caffe/layers/interp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype, bool packed>
__global__ void caffe_gpu_interp2_kernel(const int n, const float rheight, const float rwidth,
    const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      if (packed) {
  const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
  Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
  for (int c = 0; c < channels; ++c) {
    pos2[0] = pos1[0];
    pos1++;
    pos2++;
  }
      }
      else {
  const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
  Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
  for (int c = 0; c < channels; ++c) {
  pos2[0] = pos1[0];
  pos1 += Width1 * Height1;
  pos2 += Width2 * Height2;
  }
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;
    //
    if (packed) {
      const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
      Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
      for (int c = 0; c < channels; ++c) {
  pos2[0] =
    h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) + 
    h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
  pos1++;
  pos2++;
      }
    }
    else {
      const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
      Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
      for (int c = 0; c < channels; ++c) {
  pos2[0] =
    h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
    h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
  pos1 += Width1 * Height1;
  pos2 += Width2 * Height2;
      }
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  caffe_gpu_interp2_kernel<Dtype,packed><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    (num_kernels, rheight, rwidth, channels,
     data1, x1, y1, height1, width1, Height1, Width1,
     data2, x2, y2, height2, width2, Height2, Width2);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, bool packed>
__global__ void caffe_gpu_interp2_kernel_backward(const int n, const float rheight, const float rwidth,
    const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      if (packed) {
  Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
  const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
  for (int c = 0; c < channels; ++c) {
    pos1[0] += pos2[0];
    pos1++;
    pos2++;
  }
      }
      else {
  Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
  const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
  for (int c = 0; c < channels; ++c) {
    pos1[0] += pos2[0];
    pos1 += Width1 * Height1;
    pos2 += Width2 * Height2;
  }
      }
      return;
    }
    //
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    //
    const float w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Dtype w1lambda = w1r - w1;
    const Dtype w0lambda = Dtype(1.) - w1lambda;
    //
    if (packed) {
      Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
      const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
      for (int c = 0; c < channels; ++c) {
  atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
  atomicAdd(&pos1[channels * w1p], h0lambda * w1lambda * pos2[0]);
  atomicAdd(&pos1[channels * h1p * Width1], h1lambda * w0lambda * pos2[0]);
  atomicAdd(&pos1[channels * (h1p * Width1 + w1p)], h1lambda * w1lambda * pos2[0]);
  pos1++;
  pos2++;
      }
    }
    else {
      Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
      const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
      for (int c = 0; c < channels; ++c) {
  atomicAdd(&pos1[0], h0lambda * w0lambda * pos2[0]);
  atomicAdd(&pos1[w1p], h0lambda * w1lambda * pos2[0]);
  atomicAdd(&pos1[h1p * Width1], h1lambda * w0lambda * pos2[0]);
  atomicAdd(&pos1[h1p * Width1 + w1p], h1lambda * w1lambda * pos2[0]);
  pos1 += Width1 * Height1;
  pos2 += Width2 * Height2;
      }
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_interp2_backward(const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  const int num_kernels = height2 * width2;
  caffe_gpu_interp2_kernel_backward<Dtype,packed><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
    (num_kernels, rheight, rwidth, channels,
     data1, x1, y1, height1, width1, Height1, Width1,
     data2, x2, y2, height2, width2, Height2, Width2);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_gpu_interp2<Dtype,false>(num_ * channels_ * length_,
    bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_ * length_,
    bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

INSTANTIATE_LAYER_GPU_FUNCS(InterpLayer);

}  // namespace caffe
