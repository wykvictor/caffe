#include <numeric>
#include <vector>

#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename Dtype>
void UnifiedLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // put child layers' data blobs together to form the final blob
  // then fill in the label_index blob
  int shift_data = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(),
             top[0]->mutable_gpu_data() + shift_data);
    shift_data += bottom[i]->count();
  }
}

template <typename Dtype>
void UnifiedLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // copy back diff, dispatch them to child layers, scale them back
  int shift_diff = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    // dispatch diff: diff * (batch_size_sum/batch_size[i])
    caffe_copy(bottom[i]->count(), top[0]->gpu_diff() + shift_diff,
      bottom[i]->mutable_gpu_diff());
    bottom[i]->scale_diff(static_cast<Dtype>(label_index_sum_)
      / label_index_[i]);
    shift_diff += bottom[i]->count();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(UnifiedLayer);

}  // namespace caffe
