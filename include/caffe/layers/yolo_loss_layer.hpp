#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {
template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {
public:
	explicit YoloLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param) {}

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::Reshape(bottom, top);
		top[0]->ReshapeLike(*bottom[0]);
		CHECK_EQ(bottom[0]->count(), top[0]->count());
		CHECK_EQ(bottom[0]->count(1), side_*side_*(num_*(1+coords_) + classes_));
		diff_.ReshapeLike(*bottom[0]);
	}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		const YoloLossParameter& yolo_loss_param = this->layer_param_.yolo_loss_param();
		// TODO: hard code
		CHECK_EQ(yolo_loss_param.classes(), 20);
		CHECK_EQ(yolo_loss_param.coords(), 4);
		CHECK_EQ(yolo_loss_param.side(), 7);
		this->classes_ = yolo_loss_param.classes();
		this->coords_ = yolo_loss_param.coords();
		this->side_    = yolo_loss_param.side();
		this->num_     = yolo_loss_param.num();
		this->jitter_  = yolo_loss_param.jitter();
	}

	virtual inline const char* type() const { return "YoloLoss"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* label = bottom[1]->cpu_data();
		int locations = side_ * side_;

		// TODO
		// if (train)
		// bool train = true;
		// if (train) {
		// 	float avg_iou = 0;
		// 	float avg_cat = 0;
		// 	float avg_allcat = 0;
		// 	float avg_obj = 0;
		// 	float avg_anyobj = 0;
		// 	int count = 0;
		// }
		Dtype* loss = top[0]->mutable_cpu_data();
		caffe_copy(bottom[0]->count(), bottom_data, loss);
	}
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	}

	int classes_;
	int coords_;
	int side_;
	int num_;
	int jitter_;

	Blob<Dtype> diff_;
};
} // namespace caffe

#endif // CAFFE_YOLO_LOSS_LAYER_HPP_