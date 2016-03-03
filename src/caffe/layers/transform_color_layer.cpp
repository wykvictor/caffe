#include <vector>

#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransformColorLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Check top and bottom size
  CHECK_EQ(bottom.size(), top.size())
      << "Top size should be equal with bottom size.";
}

template <typename Dtype>
void TransformColorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (bottom[i]->channels() == 1) {
      top[i]->Reshape(bottom[i]->num(), 3, bottom[i]->height(),
        bottom[i]->width());
    } else if (bottom[i]->channels() > 1) {
      top[i]->Reshape(bottom[i]->num(), 1, bottom[i]->height(),
        bottom[i]->width());
    } else {
      LOG(INFO) << "channels() < 1 doesn't need transformation, copy directly";
      top[i]->CopyFrom(*bottom[i], false, true);
    }
  }
}

template <typename Dtype>
void TransformColorLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (bottom[i]->channels() == 1) {
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < 3; ++c) {
          for (int h = 0; h < bottom[i]->height(); ++h) {
            for (int w = 0; w < bottom[i]->width(); ++w) {
              top[i]->mutable_cpu_data()[top[i]->
              offset(n, c, h, w)] = bottom[i]->data_at(n, 0, h, w);
            }
          }
        }
      }
    } else if (bottom[i]->channels() > 1) {
      int bottom_channel = bottom[i]->channels();
      // copy the average of different channels to top
      for (int n = 0; n < bottom[i]->num(); ++n) {
        for (int c = 0; c < 1; ++c) {
          for (int h = 0; h < bottom[i]->height(); ++h) {
            for (int w = 0; w < bottom[i]->width(); ++w) {
              Dtype sum = static_cast<Dtype>(0);
              for (int cc = 0; cc < bottom_channel; ++cc) {
                sum += bottom[i]->data_at(n, cc, h, w);
              }
              top[i]->mutable_cpu_data()[top[i]->offset(n, c, h, w)] =
                sum / bottom_channel;
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(TransformColorLayer, Forward);
#else
template <typename Dtype>
void TransformColorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}
#endif

INSTANTIATE_CLASS(TransformColorLayer);
REGISTER_LAYER_CLASS(TransformColor);

}  // namespace caffe
