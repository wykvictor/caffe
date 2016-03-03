#ifndef CAFFE_UTILITY_LAYERS_HPP_
#define CAFFE_UTILITY_LAYERS_HPP_

#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

/**
 * @brief Merge blobs from different models into one unified blob.
 * Assumes all child layers have the same shape.
 */
template <typename Dtype>
class UnifiedLayer : public Layer<Dtype> {
 public:
  explicit UnifiedLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Unified"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  /**
  * UnifiedDataLayer need 2 top blobs: Data+Label_index
  * Label_index is used to identify different child layers
  * TODO: Remove (Label_index) from prototxt, it should be transparent to users
  */
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  // Record each data_layer's batch size
  vector<int> label_index_;
  int label_index_sum_;
  // number of child layers
  int childlayer_num_;
};

}  // namespace caffe

#endif  // CAFFE_UTILITY_LAYERS_HPP_
