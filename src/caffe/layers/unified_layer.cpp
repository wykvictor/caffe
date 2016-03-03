#include <vector>

#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename Dtype>
void UnifiedLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  childlayer_num_ = bottom.size();
  label_index_.resize(childlayer_num_);
  // blob top[1]: label_index can be reshaped here
  top[1]->Reshape(childlayer_num_, 1, 1, 1);
}

template <typename Dtype>
void UnifiedLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Reshape top blobs according to bottom blobs' num()
  label_index_sum_ = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    label_index_[i] = static_cast<int>(bottom[i]->num());
    label_index_sum_ += label_index_[i];
    // label_index can be assigned here
    top[1]->mutable_cpu_data()[i] = static_cast<Dtype>(label_index_[i]);
  }
  top[0]->Reshape(label_index_sum_, bottom[0]->channels(),
    bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void UnifiedLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // put child layers' data blobs together to form the final blob
  // then fill in the label_index blob
  int shift_data = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(),
             top[0]->mutable_cpu_data() + shift_data);
    shift_data += bottom[i]->count();
  }
}

template <typename Dtype>
void UnifiedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // copy back diff, dispatch them to child layers, scale them back
  int shift_diff = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    // dispatch diff: diff * (batch_size_sum/batch_size[i])
    caffe_copy(bottom[i]->count(), top[0]->cpu_diff() + shift_diff,
      bottom[i]->mutable_cpu_diff());
    bottom[i]->scale_diff(static_cast<Dtype>(label_index_sum_)
      / label_index_[i]);
    shift_diff += bottom[i]->count();
  }
}

#ifdef CPU_ONLY
STUB_GPU(UnifiedLayer);
#endif

INSTANTIATE_CLASS(UnifiedLayer);
REGISTER_LAYER_CLASS(Unified);

}  // namespace caffe
