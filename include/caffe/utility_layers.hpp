#ifndef CAFFE_UTILITY_LAYERS_HPP_
#define CAFFE_UTILITY_LAYERS_HPP_

#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

/**
 * @brief Before forwarding to the Net, adding this layer after date layer
 *  to force the image to have 3 color channels if it only has 1 channel now.
 *  Also, if the image data has more than 1 channels already, reduce it to 1.
 */
template <typename Dtype>
class TransformColorLayer: public Layer<Dtype> {
 public:
  explicit TransformColorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TransformColor"; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MinBottomBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  // TODO: complete the gpu version
  // Now, this layer is only used to transform mnist data
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
};

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

/**
 * @brief Dispatchs data to different top layers according to its Label_index.
 * This layer should be used with UnifiedDataLayer to merge different models.
 */
template <typename Dtype>
class DispatchLayer : public Layer<Dtype> {
 public:
  explicit DispatchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Dispatch"; }
  // Dispatch 1 bottom blob to N top blobs
  virtual inline int MinTopBlobs() const { return 1; }
  // here 2 bottom blobs contain: Data + Label_index
  virtual inline int ExactNumBottomBlobs() const { return 2; }

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
  // Record each layer's batch size, according to blob label_index(bottom[1])
  vector<int> label_index_;
  // number of child layers(models)
  int childlayer_num_;
};

}  // namespace caffe

#endif  // CAFFE_UTILITY_LAYERS_HPP_
