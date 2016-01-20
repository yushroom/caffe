#ifndef CAFFE_YOLO_LOSS_LAYER_HPP_
#define CAFFE_YOLO_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

template<typename Dtype>
class BBox
{
public:
	Dtype x;
	Dtype y;
	Dtype w;
	Dtype h;

	static BBox float_to_box(const Dtype* f);

	// Intersection over union
	static Dtype iou(const BBox& a, const BBox& b);

	static Dtype rmse(const BBox& a, const BBox& b);

private:
	static Dtype Overlap(const Dtype x1, const Dtype w1, const Dtype x2, const Dtype w2);

	// return overlap area of a and b
	static Dtype Intersection(const BBox& a, const BBox& b);

	// return total area of a and b
	static Dtype Union(const BBox& a, const BBox& b);
};

namespace caffe {

template <typename Dtype>
class YoloLossLayer : public LossLayer<Dtype> {
public:
	explicit YoloLossLayer(const LayerParameter& param);

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "YoloLoss"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int classes_;	// 20
	int coords_;	// = 4, bounding box(x, y, w, h)
	int side_;		// grid count in each row/clown, side_*side in all
	int num_;		// number of box in each grid
	int jitter_;	// for random translation

	Blob<Dtype> diff_;
};

} // namespace caffe

#endif // CAFFE_YOLO_LOSS_LAYER_HPP_