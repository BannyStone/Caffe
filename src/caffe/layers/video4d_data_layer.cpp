#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/video4d_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
Video4dDataLayer<Dtype>:: ~Video4dDataLayer<Dtype>() {
	this->JoinPrefetchThread();
}

template <typename Dtype>
void Video4dDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int new_height  = this->layer_param_.video4d_data_param().new_height();
	const int new_width  = this->layer_param_.video4d_data_param().new_width();
	const int new_length  = this->layer_param_.video4d_data_param().new_length();
	const int num_segments = this->layer_param_.video4d_data_param().num_segments();
	const int step = this->layer_param_.video4d_data_param().step(); // sampling rate along short-term time dimension
	const string& source = this->layer_param_.video4d_data_param().source();
	const bool rand_step = this->layer_param_.video4d_data_param().rand_step();
	const bool offset_share = this->layer_param_.video4d_data_param().offset_share();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label) {
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}

	if (this->layer_param_.video4d_data_param().shuffle()) {
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	//check name pattern
	if (this->layer_param_.video4d_data_param().name_pattern() == "") {
		if (this->layer_param_.video4d_data_param().modality() == Video4dDataParameter_Modality_RGB) {
			name_pattern_ = "image_%04d.jpg";
		} else if (this->layer_param_.video4d_data_param().modality() == Video4dDataParameter_Modality_FLOW)
			throw std::invalid_argument( "Video4D data layer does not support flow mdality yet." );
			// name_pattern_ = "flow_%c_%04d.jpg";
	} else {
		name_pattern_ = this->layer_param_.video4d_data_param().name_pattern();
	}

	Datum datum;
	bool is_color = !this->layer_param_.video4d_data_param().grayscale();
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_] / num_segments;
	vector<int> offsets;
	vector<vector<int> > skip_offsets;
	for (int i = 0; i < num_segments; ++i) {
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset;
		if (average_duration >= new_length) {
			offset = (*frame_rng)() % (average_duration -  new_length + 1);
		}
		else {
			offset = 0;
		}
		offsets.push_back(offset + i * average_duration);
		vector<int> tmp_off;
		for (int j=0; j< new_length; ++j) {
			// if (rand_step == true)
			// 	offset = (*frame_rng)() % step;
			// else
			offset = 0;
			tmp_off.push_back(offset);
		}
		skip_offsets.push_back(tmp_off);
	}
	if (this->layer_param_.video4d_data_param().modality() == Video4dDataParameter_Modality_FLOW)
		throw std::invalid_argument( "Video4D data layer does not support flow mdality yet." );
	else
		CHECK(ReadSegmentRGBToDatum_4D(lines_[lines_id_].first, lines_[lines_id_].second,
										offsets, new_height, new_width, new_length, &datum, 
										true, name_pattern_.c_str(), 1, skip_offsets)); // when setting up, step is restricted to 1
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video4d_data_param().batch_size();
	if (crop_size > 0) {
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void Video4dDataLayer<Dtype>::ShuffleVideos() {
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void Video4dDataLayer<Dtype>::InternalThreadEntry() {
	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	Video4dDataParameter video4d_data_param = this->layer_param_.video4d_data_param();
	const int batch_size = video4d_data_param.batch_size();
	const int new_height = video4d_data_param.new_height();
	const int new_width = video4d_data_param.new_width();
	const int new_length = video4d_data_param.new_length();
	const int num_segments = video4d_data_param.num_segments();
	const int lines_size = lines_.size();
	const int init_step = video4d_data_param.step(); // adjust step according to the size of frame sequence
	const bool rand_step = video4d_data_param.rand_step();
	const bool offset_share = video4d_data_param.offset_share();

	bool is_color = !this->layer_param_.video4d_data_param().grayscale();
	for (int item_id = 0; item_id < batch_size; ++ item_id) {
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		vector<vector<int> > skip_offsets;
		double average_duration = lines_duration_[lines_id_] / num_segments;
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		// set step dynamically
		int step = init_step;
		if (average_duration < (new_length - 1) * step + 1) {
			step = (int) (average_duration / new_length);
			if (step == 0)
				step = 1;
		}
		CHECK_GT(step, 0);
		// set offset shared between different segments.
		int share_offset;
		if (average_duration >= (new_length - 1) * step + 1)
			share_offset = (*frame_rng)() % ( (int)average_duration - (new_length - 1) * step);
		else
			share_offset = 0;
		// set offset per segment.
		for (int i = 0; i < num_segments; ++ i) {
			if (this->phase_ == TRAIN) {
				if (average_duration >= (new_length - 1) * step + 1) {
					// set offset
					int offset;
					if (!offset_share)
						offset = (*frame_rng)() % ( (int)average_duration - (new_length - 1) * step);
					else
						offset = share_offset;
					offsets.push_back( (int) (offset + i * average_duration));
					vector<int> tmp_off;
					for (int j = 0; j < new_length; ++ j) {
						if (rand_step == true)
							offset = (*frame_rng)() % step;
						else
							offset = 0;
						tmp_off.push_back(offset);
					}
					skip_offsets.push_back(tmp_off);
				} else {
					CHECK_EQ(step, 1); // After above operations, step must be one here.
					offsets.push_back((int) (i * average_duration));
					vector<int> tmp_off;
					for (int j = 0; j < new_length; j++) {
						tmp_off.push_back(0);
					}
					skip_offsets.push_back(tmp_off);
				}
			} else {
				if (average_duration >= new_length) {
					if (average_duration < (new_length - 1) * step + 1) {
						step = (int) (average_duration / new_length);
						CHECK_GT(step, 0); // because average_duration >= new_length
					}
				    offsets.push_back(int((average_duration - (new_length - 1) * step) / 2 +
				    						i * average_duration));
				} else {
					step = 1;
				    offsets.push_back((int) (i * average_duration));
				}
				vector<int> tmp_off;
				for (int j = 0; j < new_length; j++)
					tmp_off.push_back(0);
				skip_offsets.push_back(tmp_off);
			}
		}
		if (this->layer_param_.video4d_data_param().modality() == Video4dDataParameter_Modality_FLOW)
			throw std::invalid_argument( "Video4D data layer does not support flow mdality yet." );
		else
			if(!ReadSegmentRGBToDatum_4D(lines_[lines_id_].first, lines_[lines_id_].second,
										offsets, new_height, new_width, new_length, &datum, 
										true, name_pattern_.c_str(), step, skip_offsets)) {
				LOG(INFO) << "Failed to read 4d data for video " << lines_[lines_id_].first;
				continue;
			}

		int offset1 = this->prefetch_data_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		top_label[item_id] = lines_[lines_id_].second;

		//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video4d_data_param().shuffle()) {
				ShuffleVideos();
			}
		}
	}
}

INSTANTIATE_CLASS(Video4dDataLayer);
REGISTER_LAYER_CLASS(Video4dData);
}
