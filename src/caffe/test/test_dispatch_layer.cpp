#include <vector>

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename TypeParam>
class DispatchLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DispatchLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_index_(new Blob<Dtype>()),
        blob_top_data_1_(new Blob<Dtype>()),
        blob_top_data_2_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    // use 2 child layers to do test, one batch_size is 2, another is 3
    blob_bottom_label_index_->Reshape(2, 1, 1, 1);
    blob_bottom_label_index_->mutable_cpu_data()[0] = static_cast<Dtype>(2);
    blob_bottom_label_index_->mutable_cpu_data()[1] = static_cast<Dtype>(3);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_index_);
    blob_top_vec_.push_back(blob_top_data_1_);
    blob_top_vec_.push_back(blob_top_data_2_);
  }

  virtual void FillBottomData() {
    // fill bottom blobs to test Forward()
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_bottom_data_->Reshape(5, 3, 6, 4);
    filler.Fill(this->blob_bottom_data_);
  }

  virtual void FillTopData() {
    // fill top blobs to test Backward()
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_top_data_1_->Reshape(2, 3, 6, 4);
    filler.Fill(this->blob_top_data_1_);
    this->blob_top_data_2_->Reshape(3, 3, 6, 4);
    filler.Fill(this->blob_top_data_2_);
    // backward() only back-propagate diff
    // so copy data to diff, otherwise diff will be 0
    caffe_copy(this->blob_top_data_1_->count(),
      this->blob_top_data_1_->cpu_data(),
      static_cast<Dtype*>(this->blob_top_data_1_->mutable_cpu_diff()));
    caffe_copy(this->blob_top_data_2_->count(),
      this->blob_top_data_2_->cpu_data(),
      static_cast<Dtype*>(this->blob_top_data_2_->mutable_cpu_diff()));
  }

  virtual ~DispatchLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_index_;
    delete blob_top_data_1_;
    delete blob_top_data_2_;
  }

  // dispatch 1 unified_layer to 2 child layers
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_index_;
  Blob<Dtype>* const blob_top_data_1_;
  Blob<Dtype>* const blob_top_data_2_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DispatchLayerTest, TestDtypesAndDevices);

TYPED_TEST(DispatchLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new DispatchLayer<Dtype>(layer_param));
  this->FillBottomData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_1_->num(), 2);
  EXPECT_EQ(this->blob_top_data_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_1_->height(), 6);
  EXPECT_EQ(this->blob_top_data_1_->width(), 4);

  EXPECT_EQ(this->blob_top_data_2_->num(), 3);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_2_->height(), 6);
  EXPECT_EQ(this->blob_top_data_2_->width(), 4);
}

TYPED_TEST(DispatchLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new DispatchLayer<Dtype>(layer_param));
  this->FillBottomData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call forward and check result
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test Label_index: batch_size are 2 and 3 respectively
  EXPECT_EQ(this->blob_top_data_1_->num(), 2);
  EXPECT_EQ(this->blob_top_data_2_->num(), 3);
  // Test Data
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < this->blob_top_data_1_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_data_1_->height(); ++h) {
        for (int w = 0; w < this->blob_top_data_1_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_data_->data_at(n, c, h, w),
           this->blob_top_data_1_->data_at(n, c, h, w));
        }
      }
    }
  }
  for (int n = 0; n < 3; ++n) {
    for (int c = 0; c < this->blob_top_data_2_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_data_2_->height(); ++h) {
        for (int w = 0; w < this->blob_top_data_2_->width(); ++w) {
          EXPECT_EQ(this->blob_bottom_data_->data_at(n + 2, c, h, w),
           this->blob_top_data_2_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(DispatchLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new DispatchLayer<Dtype>(layer_param));
  // call fillBottomData first to shape the bottom blob
  this->FillBottomData();
  this->FillTopData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call backward and check result
  // use fake propagate_down only, for it's useless in DispatchLayer->Backward
  layer->Backward(this->blob_top_vec_, *new vector<bool>(),
    this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_data_->num(), 5);
  EXPECT_EQ(this->blob_bottom_data_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_data_->height(), 6);
  EXPECT_EQ(this->blob_bottom_data_->width(), 4);
  // Test Diff
  for (int n = 0; n < 5; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 6; ++h) {
        for (int w = 0; w < 4; ++w) {
          EXPECT_EQ(this->blob_bottom_data_->diff_at(n, c, h, w), n < 2 ?
            static_cast<Dtype>(2) / 5 * this->blob_top_data_1_->
            diff_at(n, c, h, w)
            : static_cast<Dtype>(3) / 5 * this->blob_top_data_2_->
            diff_at(n - 2, c, h, w));
        }
      }
    }
  }
}

}  // namespace caffe
