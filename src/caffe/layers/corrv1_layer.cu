#include <vector>
#include <stdio.h>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/corrv1_layer.hpp"

#include "caffe/caffe.hpp"

#define ROUND_OFF 50000

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

namespace caffe {

// == Dimension rearrangement Kernel
// <<<[(bwidthheight-1)/threads_per_block+1, bchannels, bnum], 16>>>
template <typename Dtype>
__global__ void template_rearrange_kernel1(const Dtype* in, Dtype* out, int template_index, 
                                          int channels, int length, int width, 
                                          int height, int widthheight, int padding, int pwidthheight) {

  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  if(xy >= widthheight)
    return;

  int ch = blockIdx.y;
  int n  = blockIdx.z;

  Dtype value=in[((n * channels + ch) * length + template_index) * widthheight + xy];

  __syncthreads();

  int xpad  = (xy % width) + padding;
  int ypad  = (xy / width) % height + padding;
  int xypad = ypad * (width + 2 * padding) + xpad;

  out[(n * pwidthheight + xypad) * channels + ch] = value;
}

// <<<[(bwidthheight-1)/threads_per_block+1, bchannels, bnum], 16>>>
template <typename Dtype>
__global__ void template_rearrange_kernel2(const Dtype* in, Dtype* out, int channels, 
  int length, int width, int height, int widthheight, int padding, int pwidthheight) {

  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  if(xy >= widthheight)
    return;

  int ch = blockIdx.y;
  int n  = blockIdx.z;

  Dtype value = 0;
  for (int t = 0; t < length; t++)
    value += in[((n * channels + ch) * length + t) * widthheight + xy];

  __syncthreads();

  int xpad = xy % width + padding;
  int ypad = (xy / width) % height + padding;
  int xypad = ypad * (width + 2 * padding) + xpad;

  out[(n*pwidthheight+xypad) * channels + ch] = value/length;
}

template <typename Dtype>
__global__ void volume_rearrange_kernel(const Dtype* in, Dtype* out, int num, int channels, 
  int length, int width, int height, int widthheight, int padding, int pwidthheight) {
  
  int xyt = blockIdx.x * blockDim.x + threadIdx.x;
  if(xyt>=widthheight*length)
    return;

  int ch = blockIdx.y;
  int n  = blockIdx.z;

  Dtype value=in[(n * channels + ch) * length * widthheight + xyt];

  __syncthreads();

  int xpad = xyt % width + padding;
  int ypad = (xyt / width) % height + padding;
  int t = (xyt / widthheight) % length; // = (xyt / widthheight)
  int xytpad = (t * (height + 2 * padding) + ypad) * (width + 2 * padding) + xpad;

  out[(n*length*pwidthheight+xytpad)*channels + ch] = value;
}

// == Correlation Kernel <<<(top_width, top_height, num*length), 32, kernel_size*kernel_size*bottomchannels>>>
// nthreads: topcount = top_length_ * top_width_ * top_height_ * top_channels_;
template <typename Dtype>
__global__ void CorrelateForward(int length, int topwidth, int topheight, 
  int topchannels, int topcount, int max_displacement, int neighborhood_grid_radius, 
  int neighborhood_grid_width, int kernel_radius, int kernel_size, 
  int stride1, int stride2, int bottomwidth, int bottomheight, int bottomchannels,
  const Dtype *corr_template, const Dtype *bottom, Dtype *top) {
  
  extern __shared__ char patch_data_char[];
  
  Dtype *patch_data = (Dtype *)patch_data_char;

  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in template image
  int x1 = blockIdx.x*stride1 + max_displacement;
  int y1 = blockIdx.y*stride1 + max_displacement;
  int item_t = blockIdx.z;
  int ch_off = threadIdx.x;
  int temporal_index = item_t % length;
  int item = item_t / length;
  
  // Load 3D patch into shared shared memory
  for (int j = 0; j < kernel_size; j++) { // HEIGHT
    for (int i = 0; i < kernel_size; i++) { // WIDTH
      int ji_off = (j * kernel_size + i) * bottomchannels;
      for (int ch = ch_off; ch < bottomchannels; ch += WARPS_PER_BLOCK * THREADS_PER_WARP) { // CHANNELS
          int idx1 = ((item * bottomheight + y1 + j) * bottomwidth + x1 + i) * bottomchannels + ch;
          int idxPatchData = ji_off + ch;
          patch_data[idxPatchData] = corr_template[idx1];
      }
    }
  }
  
  __syncthreads();
  
  __shared__ Dtype sum[WARPS_PER_BLOCK * THREADS_PER_WARP];
  // Dtype sum[WARPS_PER_BLOCK * THREADS_PER_WARP];
  
  // Compute correlation (No speed acceleration yet)
  int offset = (item * length + temporal_index) * bottomheight * bottomwidth * bottomchannels;
  for (int top_channel = 0; top_channel < topchannels; top_channel++) {
    sum[ch_off] = 0;

    int s2o = (top_channel % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    int s2p = (top_channel / neighborhood_grid_width - neighborhood_grid_radius) * stride2;
    
    for(int j = 0; j < kernel_size; j++) { // HEIGHT
      for(int i = 0; i < kernel_size; i++) { // WIDTH
        int ji_off = (j * kernel_size + i) * bottomchannels;
        for(int ch = ch_off; ch < bottomchannels; ch += WARPS_PER_BLOCK * THREADS_PER_WARP) { // CHANNELS
          int x2 = x1 + s2o;
          int y2 = y1 + s2p;
          
          int idxPatchData = ji_off + ch;
          // int idx2 = ((item * bottomheight + y2+j) * bottomwidth + x2+i) * bottomchannels + ch;
          int idx2 = offset + ((y2 + j) * bottomwidth + x2 + i) * bottomchannels + ch;
          
          sum[ch_off] += patch_data[idxPatchData] * bottom[idx2];
        }
      }
    }
    
    __syncthreads();
    
    if(ch_off == 0) {
        Dtype total_sum = 0;
        for(int idx = 0; idx < WARPS_PER_BLOCK * THREADS_PER_WARP; idx++) {
            total_sum += sum[idx];
        }
        const int sumelems = kernel_size * kernel_size * bottomchannels;
        const int index = (((top_channel * length + temporal_index) * topheight + blockIdx.y) * topwidth) + blockIdx.x;
        top[index + item*topcount] = total_sum / (float)sumelems;
    }
  }
  // Aggregate
}

// == Correlation Backward Pass Kernel (For Blob 0)
template <typename Dtype>
__global__ void CorrelateTemplateBackward(int template_index, int length,
  int topwidth, int topheight, int topchannels, int max_displacement, int neighborhood_grid_radius, 
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2, int bottomwidth, 
  int bottomheight, int bottomwidthheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, 
  int pad_size, Dtype *bottomdiff, const Dtype *bottom, const Dtype *topdiff) {

  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  if (xy >= bottomwidthheight)
    return;
  int l = xy % bottomwidth + pad_size; //w-pos
  int m = xy / bottomwidth + pad_size; //h-pos

  int n = blockIdx.y; //channel
  int item = blockIdx.z; //batch index

  //Get X,Y ranges and clamp
  // round_off is a trick to enable integer division with ceil, even for negative numbers
  // We use a large offset, for the inner part not to become negative.
  const int round_off = ROUND_OFF;
  const int round_off_s1 = stride1 * round_off;
  
  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
  int xmin = (l - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
  int ymin = (m - 2*kernel_radius - max_displacement + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement) / stride1
  
  // Same here:
  int xmax = (l - max_displacement + round_off_s1) / stride1 - round_off; // floor (l - max_displacement) / stride1
  int ymax = (m - max_displacement + round_off_s1) / stride1 - round_off; // floor (m - max_displacement) / stride1

  Dtype sum = 0;
  for (int t = 0; t < length; t++) {
    if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1)) {

      xmin = max(0,xmin);
      xmax = min(topwidth-1,xmax);

      ymin = max(0,ymin);
      ymax = min(topheight-1,ymax);

      for(int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
        for(int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {

          // Get bottom1 data:
          int s2o = stride2 * o;
          int s2p = stride2 * p;
          int idxbot1 = ((item * pbottomheight + (m+s2p)) * pbottomwidth + (l+s2o)) * bottomchannels + n;
          Dtype bot1tmp = bottom[idxbot1]; // bottom1[l+s2o,m+s2p,n]

          // Index offset for topdiff in following loops:
          int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
          int idxopoffset = (item * topchannels + op);

          for(int y = ymin; y <= ymax; y++) {
            for(int x = xmin; x <= xmax; x++) {
              int idxtopdiff = (idxopoffset * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
              sum += topdiff[idxtopdiff] * bot1tmp;
            }
          }
        }
      }
    }
  }

  int bot0index;
  const int sumelems = length * (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bottomchannels;
  if (template_index != -1) {
    bot0index = (((n * length + template_index) * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
    bottomdiff[bot0index + item * bottomcount] += sum / (float)sumelems;
  } else {
    for (int t = 0; t < length; t++) {
      bot0index = (((n * length + t) * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
      bottomdiff[bot0index + item * bottomcount] += sum / ((float)sumelems * length);
    }
  }
}

// == Correlation Backward Pass Kernel (No speed optimization yet)
// nthreads: bottomcount = channels * length * height * width
template <typename Dtype>
__global__ void CorrelateVolumeBackward(int length, int topwidth, 
  int topheight, int topchannels, int max_displacement, int neighborhood_grid_radius, 
  int neighborhood_grid_width, int kernel_radius, int stride1, int stride2, int bottomwidth, 
  int bottomheight, int bottomwidthheight, int pbottomwidth, int pbottomheight, int bottomchannels, int bottomcount, 
  int pad_size, const Dtype *corr_template, Dtype *bottomdiff, const Dtype *topdiff) {
  
  int xy = blockIdx.x * blockDim.x + threadIdx.x;

  if (xy >= bottomwidthheight)
    return;

  int l = (xy % bottomwidth) + pad_size; //w-pos
  int m = (xy / bottomwidth) + pad_size; //h-pos

  int n = blockIdx.y; //channel
  int t = blockIdx.z % length; //t-pos
  int item = blockIdx.z / length; //batch index
  
  // round_off is a trick to enable integer division with ceil, even for negative numbers
  // We use a large offset, for the inner part not to become negative.
  const int round_off = ROUND_OFF;
  const int round_off_s1 = stride1 * round_off;
  
  Dtype sum = 0;
  for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
    for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius; o++) {
      
      int s2o = stride2 * o;
      int s2p = stride2 * p;
      
      //Get X,Y ranges and clamp
      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
      int xmin = (l - 2*kernel_radius - max_displacement - s2o + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
      int ymin = (m - 2*kernel_radius - max_displacement - s2p + round_off_s1 - 1) / stride1 + 1 - round_off; // ceil (l - 2*kernel_radius - max_displacement - s2o) / stride1
      
      // Same here:
      int xmax = (l - max_displacement - s2o + round_off_s1) / stride1 - round_off; // floor (l - max_displacement - s2o) / stride1
      int ymax = (m - max_displacement - s2p + round_off_s1) / stride1 - round_off; // floor (m - max_displacement - s2p) / stride1

      if(xmax>=0 && ymax>=0 && (xmin<=topwidth-1) && (ymin<=topheight-1))
      {
        xmin = max(0,xmin);
        xmax = min(topwidth-1,xmax);

        ymin = max(0,ymin);
        ymax = min(topheight-1,ymax);

        // Get template data:
        int idxbot0 = ((item * pbottomheight + (m - s2p)) * pbottomwidth + (l - s2o)) * bottomchannels + n;
        Dtype bot0tmp = corr_template[idxbot0]; // template[l-s2o, m-s2p, n]

        // Index offset for topdiff in following loops:
        int op = (p+neighborhood_grid_radius) * neighborhood_grid_width + (o+neighborhood_grid_radius); // index [o,p]
        int idxOpOffset = (item * topchannels + op);

        for(int y = ymin; y <= ymax; y++) {
          for(int x = xmin; x <= xmax; x++) {
            int idxtopdiff = ((idxOpOffset * length + t) * topheight + y) * topwidth + x; // topdiff[x,y,o,p]
            sum += topdiff[idxtopdiff] * bot0tmp;
          }
        }
      }
    }
  }

  const int sumelems = (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bottomchannels;
  const int bot1index = (((n * length + t) * bottomheight) + (m - pad_size)) * bottomwidth + (l - pad_size);
  bottomdiff[bot1index + item * bottomcount] += sum / (float)sumelems;
}

// == Forward
template <typename Dtype>
void Corrv1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);

  const int bnum = bottom[0]->shape(0);
  const int bchannels = bottom[0]->shape(1);
  const int blength = bottom[0]->shape(2);
  const int bheight = bottom[0]->shape(3);
  const int bwidth = bottom[0]->shape(4);

  const int bwidthheight = bwidth * bheight;

  const int topcount = top_length_ * top_width_ * top_height_ * top_channels_;

  dim3 threadsPerBlock(THREADS_PER_WARP * WARPS_PER_BLOCK);

  cudaMemset(rbot_->mutable_gpu_data(), 0, rbot_->count()*sizeof(Dtype));

  int threads_per_block=16;
  dim3 totalBlocksTemplateRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
  dim3 totalBlocksVolumeRearr((bwidthheight*blength-1)/threads_per_block+1, bchannels, bnum);
  const int pwidthheight = (bwidth + 2 * pad_size_) * (bheight + 2 * pad_size_);

  if (template_type_ == Corrv1Parameter_TemplateType_INDEX) {
    template_rearrange_kernel1<<<totalBlocksTemplateRearr, threads_per_block>>>
    (bottom[0]->gpu_data(), template_->mutable_gpu_data(), template_index_, 
      bchannels, blength, bwidth,
      bheight, bwidthheight, pad_size_, pwidthheight);
  } else if (template_type_ == Corrv1Parameter_TemplateType_SUM) {
    template_rearrange_kernel2<<<totalBlocksTemplateRearr, threads_per_block>>>
    (bottom[0]->gpu_data(), template_->mutable_gpu_data(), 
      bchannels, blength, bwidth, 
      bheight, bwidthheight, pad_size_, pwidthheight);
  } else {
    NOT_IMPLEMENTED;
  }

  // [2,9,9,3]
  // printf("template[0,0,0,0]: %f 0\n", template_->cpu_data()[0]);
  // printf("template[0,1,1,0]: %f 1\n", template_->cpu_data()[30]);
  // printf("template[0,7,7,0]: %f 1\n", template_->cpu_data()[210]);
  
  volume_rearrange_kernel<Dtype><<<totalBlocksVolumeRearr, threads_per_block>>>
  (bottom[0]->gpu_data(), rbot_->mutable_gpu_data(), bnum, bchannels,
    blength, bwidth, bheight, bwidthheight, pad_size_, pwidthheight);

  // [2,3,9,9,3]
  // printf("rbot[0,0,0,0,1]: %f 0\n", rbot_->cpu_data()[1]);
  // printf("rbot[0,2,0,8,1]: %f 0\n", rbot_->cpu_data()[24+81*3*2+1]);
  // printf("rbot[0,0,1,1,1]: %f 1\n", rbot_->cpu_data()[31]);
  // printf("rbot[0,1,1,1,1]: %f 1\n", rbot_->cpu_data()[30+81*3+1]);

  const int num = bnum;
  const int channels = bchannels;
  const int height = bheight + 2*pad_size_;
  const int width = bwidth + 2*pad_size_;

  const int shared_memory_per_block = (kernel_size_*kernel_size_)*bchannels;

  if(corr_type_ == Corrv1Parameter_CorrelationType_MULTIPLY) {
    // int topThreadCount = topcount;

    dim3 totalBlocksCorr(top_width_, top_height_, num * top_length_);

    CorrelateForward<Dtype><<<totalBlocksCorr, threadsPerBlock, shared_memory_per_block * sizeof(Dtype)>>>(
      blength, top_width_, top_height_, 
      top_channels_, topcount, max_displacement_, neighborhood_grid_radius_, 
      neighborhood_grid_width_, kernel_radius_, kernel_size_, 
      stride1_, stride2_, width, height, channels,
      template_->gpu_data(), rbot_->gpu_data(), top[0]->mutable_gpu_data());

    CUDA_POST_KERNEL_CHECK;

  } else {
    NOT_IMPLEMENTED;
  }
}

// == Backward
template <typename Dtype>
void Corrv1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Get top diff, compute bottom diff
  const Dtype* top_diff = top[0]->gpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  cudaMemset(bottom_diff, 0, bottom[0]->count()*sizeof(Dtype));

  const int num = bottom[0]->shape(0);
  const int channels = bottom[0]->shape(1);
  const int length = bottom[0]->shape(2);
  const int height = bottom[0]->shape(3);
  const int width = bottom[0]->shape(4);

  const int bottomwidthheight = width * height;
  const int paddedheight = height + 2 * pad_size_;
  const int paddedwidth = width + 2 * pad_size_;
  const int bottomcount = channels * length * height * width;

  int threads_per_block=16;
  dim3 totalBlocksVolumeBack((bottomwidthheight - 1) / threads_per_block + 1, channels, num * length);
  dim3 totalBlocksTemplateBack((bottomwidthheight - 1) / threads_per_block + 1, channels, num);

  if(corr_type_ == Corrv1Parameter_CorrelationType_MULTIPLY) {

    // == Run kernel Volume Backward
    CorrelateVolumeBackward<Dtype><<<totalBlocksVolumeBack, threads_per_block>>>(
      length, top_width_, 
      top_height_, top_channels_, max_displacement_, neighborhood_grid_radius_, 
      neighborhood_grid_width_, kernel_radius_, stride1_, stride2_, width, 
      height, bottomwidthheight, paddedwidth, paddedheight, channels, bottomcount, 
      pad_size_, template_->gpu_data(), bottom_diff, top_diff);

    CUDA_POST_KERNEL_CHECK;

    CorrelateTemplateBackward<Dtype><<<totalBlocksTemplateBack, threads_per_block>>>(
      template_index_, length,
      top_width_, top_height_, top_channels_, max_displacement_, neighborhood_grid_radius_, 
      neighborhood_grid_width_, kernel_radius_, stride1_, stride2_, width,
      height, bottomwidthheight, paddedwidth, paddedheight, channels, bottomcount,
      pad_size_, bottom_diff, rbot_->gpu_data(), top_diff);

    CUDA_POST_KERNEL_CHECK;
  } else {
    NOT_IMPLEMENTED;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(Corrv1Layer);

}  // namespace caffe
