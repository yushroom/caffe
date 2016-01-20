#include "caffe/layers/yolo_loss_layer.hpp"

template<typename Dtype>
BBox<Dtype> BBox<Dtype>::float_to_box(const Dtype* f) {
	BBox b;
	b.x = f[0];
	b.y = f[1];
	b.w = f[2];
	b.h = f[3];
	return b;
}

template<typename Dtype>
Dtype BBox<Dtype>::iou(const BBox<Dtype>& a, const BBox<Dtype>& b) {
	return Intersection(a, b) / Union(a, b);
}

template<typename Dtype>
Dtype BBox<Dtype>::rmse(const BBox<Dtype>& a, const BBox<Dtype>& b) {
	return sqrt(pow(a.x-b.x, 2) + pow(a.y-b.y, 2) + pow(a.w-b.w, 2) + pow(a.h-b.h, 2));
}

template<typename Dtype>
Dtype BBox<Dtype>::Overlap(const Dtype x1, const Dtype w1, const Dtype x2, const Dtype w2) {
	Dtype l1 = x1 - w1/2;
	Dtype l2 = x2 - w2/2;
	Dtype left = l1 > l2 ? l1 : l2;
	Dtype r1 = x1 + w1/2;
	Dtype r2 = x2 + w2/2;
	Dtype right = r1 < r2 ? r1 : r2;
	return right - left;
}

// reatun overlap area of a and b
template<typename Dtype>
Dtype BBox<Dtype>::Intersection(const BBox<Dtype>& a, const BBox<Dtype>& b) {
	Dtype w = Overlap(a.x, a.w, b.x, b.w);
	Dtype h = Overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	return w * h;
}

template<typename Dtype>
Dtype BBox<Dtype>::Union(const BBox<Dtype>& a, const BBox<Dtype>& b) {
	Dtype interset = Intersection(a, b);
	return  a.w*a.h + b.w*b.h - interset;
}

namespace caffe {

template <typename Dtype>
YoloLossLayer<Dtype>::YoloLossLayer(const LayerParameter& param)
	: LossLayer<Dtype>(param) {

}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	top[0]->ReshapeLike(*bottom[0]);
	CHECK_EQ(bottom[0]->count(), top[0]->count());
	CHECK_EQ(bottom[0]->count(1), side_*side_*(num_*(1+coords_) + classes_));
	CHECK_EQ(bottom[1]->count(1), side_*side_*(1+coords_+classes_));
	// check bacth
	CHECK_EQ(bottom[0]->count(0)/bottom[1]->count(1), bottom[1]->count(0)/bottom[1]->count(1));
	//LOG(FATAL) << bottom[1]->count(0) << ' ' << bottom[1]->count(2) << ' ' << bottom[1]->count(2);
	diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	const YoloLossParameter& yolo_loss_param = this->layer_param_.yolo_loss_param();
	// TODO: hard code
	CHECK_EQ(yolo_loss_param.classes(), 20);
	CHECK_EQ(yolo_loss_param.coords(), 4);
	CHECK_EQ(yolo_loss_param.side(), 7);
	this->classes_ = yolo_loss_param.classes();
	this->coords_  = yolo_loss_param.coords();
	this->side_    = yolo_loss_param.side();
	this->num_     = yolo_loss_param.num();
	this->jitter_  = yolo_loss_param.jitter();
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	//Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();
	Dtype* loss = top[0]->mutable_cpu_data();
	caffe_copy(bottom[0]->count(), bottom_data, loss);

	Dtype* p_diff = diff_.mutable_cpu_data();

	int bottom_count = bottom[0]->count();
	int inputs = bottom_count / bottom[0]->count(1);
	int locations = side_ * side_; 	// num_grid
	int num_batch = bottom[0]->count(0)/bottom[1]->count(1);

	const float object_scale=1;
	const float noobject_scale=.5f;
	const float class_scale=1;
	const float coord_scale=5;
	const bool use_sqrt = true;
	const bool rescore = true;
	// TODO
	bool train = true;
	caffe_memset(bottom_count, 0, diff_.mutable_cpu_data());
	if (train) {
		float avg_iou = 0;
		float avg_cat = 0;
		float avg_allcat = 0;
		float avg_obj = 0;
		float avg_anyobj = 0;
		float cost = 0;
		int count = 0;
		for (int batch_idx = 0; batch_idx < num_batch; ++batch_idx) {	// for each image in batch
			int start_index = batch_idx * inputs;
			for (int grid_idx = 0; grid_idx < locations; ++grid_idx) {
				int truth_index = (batch_idx*locations + grid_idx) * (1+coords_+classes_);
				int is_obj = label[truth_index];
				for (int j = 0; j < num_; ++j) {
					int p_index = start_index + locations*classes_ + grid_idx*num_ + j;
					p_diff[p_index] = noobject_scale * (0 - bottom_data[p_index]);
					cost += noobject_scale*pow(bottom_data[p_index], 2);
					avg_anyobj += bottom_data[p_index];
				}

				if (is_obj == 0) {
					continue;
				}

				int best_index = -1;
				float best_iou = 0;
				float best_rmse = 20;

				int class_index = start_index + grid_idx * this->classes_;
				for (int j = 0; j < classes_; ++j) {
					p_diff[class_index+j] = class_scale * (label[truth_index+1+j]) -
											bottom_data[class_index + j];
					cost += class_scale * pow(label[truth_index+1+j] - bottom_data[class_index+j], 2);
					if (label[truth_index + 1 + j])
						avg_cat += bottom_data[class_index+j];
					avg_allcat += bottom_data[class_index+j];
				}

				BBox<Dtype> truth = BBox<Dtype>::float_to_box(label + truth_index + 1 + this->classes_);
				truth.x /= side_;
				truth.y /= side_;

				for (int j = 0; j < num_; ++j) {
					int box_index = start_index + locations * (classes_ + num_) +
									(grid_idx * num_ + j) * coords_;
					BBox<Dtype> out = BBox<Dtype>::float_to_box(bottom_data + box_index);
					out.x /= side_;
					out.y /= side_;

					if (use_sqrt) {
						out.w *= out.w;
						out.h *= out.h;
					}

					float iou = BBox<Dtype>::iou(out, truth);
					float rmse = BBox<Dtype>::rmse(out, truth);
					if (best_iou > 0 || iou > 0) {
						if (iou > best_iou) {
							best_iou = iou;
							best_index = j;
						}
					} else {
						if (rmse < best_rmse) {
							best_rmse = rmse;
							best_index = j;
						}
					}
				}

				// if (l.forced) {
				// }

				int box_index = start_index + locations*(classes_+this->num_) + (grid_idx*num_+best_index)*coords_;
				int tbox_index = truth_index + 1 + classes_;

				BBox<Dtype> out = BBox<Dtype>::float_to_box(bottom_data + box_index);
				out.x /= side_;
				out.y /= side_;
				if (use_sqrt) {
					out.w *= out.w;
					out.h *= out.h;
				}
				float iou = BBox<Dtype>::iou(out, truth);
				LOG(INFO) << "best index: " << best_index;
				int p_index = start_index + locations*classes_ + grid_idx*num_ + best_index;
				cost -= noobject_scale * pow(bottom_data[p_index], 2);
				cost += object_scale * pow(1-bottom_data[p_index], 2);
				avg_obj += bottom_data[p_index];
				p_diff[p_index] = object_scale * (1.f - bottom_data[p_index]);

				if (rescore) {
					p_diff[p_index] = object_scale * (iou - bottom_data[p_index]);
				}

				p_diff[box_index+0] = coord_scale * (label[tbox_index+0] - bottom_data[box_index + 0]);
				p_diff[box_index+1] = coord_scale * (label[tbox_index+1] - bottom_data[box_index + 1]);
				if (use_sqrt) {
					p_diff[box_index+2] = coord_scale * (sqrt(label[tbox_index+2]) - bottom_data[box_index + 2]);
					p_diff[box_index+3] = coord_scale * (sqrt(label[tbox_index+3]) - bottom_data[box_index + 3]);	
				} else {
					p_diff[box_index+2] = coord_scale * (label[tbox_index+2] - bottom_data[box_index + 2]);
					p_diff[box_index+3] = coord_scale * (label[tbox_index+3] - bottom_data[box_index + 3]);
				}

				cost += pow(1-iou, 2);
				avg_iou += iou;
				++count;
			}
			printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", 
				avg_iou/count, avg_cat/count, avg_allcat/(count*classes_), avg_obj/count, avg_anyobj/(num_batch*locations*num_), count);
		}
	}
	// caffe_sub(
	// 	bottom[0]->count(),
	// 	bottom[0]->cpu_data(),
	// 	bottom[1]->cpu_data(),
	// 	diff_.mutable_cpu_data());
	
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1]) {
		LOG(FATAL)  << this->type()
					<< " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		//Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		//const Dtype* label = bottom[1]->cpu_data();
		caffe_axpy( bottom[0]->count(),				// N
					Dtype(1),						// alpha
					diff_.cpu_data(),				// X
		 			bottom[0]->mutable_cpu_diff());	// Y
	}
}

INSTANTIATE_CLASS(YoloLossLayer);
REGISTER_LAYER_CLASS(YoloLoss);
}