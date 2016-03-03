#include <numeric>
#include <vector>

#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename Dtype>
void DispatchLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Dispatch the unified data/label blob to different top models
  int shift_data = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    caffe_copy(top[i]->count(), bottom[0]->gpu_data() + shift_data,
              top[i]->mutable_gpu_data());
    shift_data += top[i]->count();
  }
}

template <typename Dtype>
void DispatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // calculate the batch_size_sum
  int label_index_sum_ = bottom[0]->num();
  // copy back diff and merge them together
  int shift_diff = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    // merge diff: diff * (batch_size[i]/batch_size_sum)
    // it's better not modify the top blob, so call scal() after copy()
    // top[i]->scale_diff(static_cast<Dtype>(label_index_[i]) /
    // label_index_sum_);
    caffe_copy(top[i]->count(), top[i]->gpu_diff(),
      bottom[0]->mutable_gpu_diff() + shift_diff);
    caffe_gpu_scal(top[i]->count(),
      static_cast<Dtype>(label_index_[i]) / label_index_sum_,
      bottom[0]->mutable_gpu_diff() + shift_diff);
    shift_diff += top[i]->count();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DispatchLayer);

}  // namespace caffe
