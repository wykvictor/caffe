#include <vector>

#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename Dtype>
void DispatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  childlayer_num_ = top.size();
  // Top size is the model num, should be equal with label_index->num()
  CHECK_EQ(childlayer_num_, bottom[1]->num())
      << "Top size does not match the number of models given by label_index";
  label_index_.resize(childlayer_num_);
}

template <typename Dtype>
void DispatchLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Reshape top blobs according to label_index(assigned in unified_layer)
  for (int i = 0; i < childlayer_num_; ++i) {
    label_index_[i] = static_cast<int>(bottom[1]->data_at(i, 0, 0, 0));
    top[i]->Reshape(label_index_[i], bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
void DispatchLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Dispatch the unified data/label blob to different top models
  int shift_data = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    caffe_copy(top[i]->count(), bottom[0]->cpu_data() + shift_data,
              top[i]->mutable_cpu_data());
    shift_data += top[i]->count();
  }
}

template <typename Dtype>
void DispatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Maybe don't need to check propagate_down
  // we always need backward to merge different top blobs to bottom layers
  // if(propagate_down[0] ) {
  //   return;
  // }

  // the whole batch_size
  int label_index_sum_ = bottom[0]->num();
  // copy back diff and merge them together
  int shift_diff = 0;
  for (int i = 0; i < childlayer_num_; ++i) {
    // merge diff: diff * (batch_size[i]/batch_size_sum)
    // it's better not modify the top blob, so call scal() after copy()
    // top[j]->scale_diff(static_cast<Dtype>(label_index_[i]) /
    // label_index_sum_);
    caffe_copy(top[i]->count(), top[i]->cpu_diff(),
      bottom[0]->mutable_cpu_diff() + shift_diff);
    caffe_scal(top[i]->count(),
      static_cast<Dtype>(label_index_[i]) / label_index_sum_,
      bottom[0]->mutable_cpu_diff() + shift_diff);
    shift_diff += top[i]->count();
  }
}

#ifdef CPU_ONLY
STUB_GPU(DispatchLayer);
#endif

INSTANTIATE_CLASS(DispatchLayer);
REGISTER_LAYER_CLASS(Dispatch);

}  // namespace caffe
