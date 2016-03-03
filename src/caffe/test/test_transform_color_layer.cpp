#include <vector>

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename TypeParam>
class TransformColorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TransformColorLayerTest()
      : blob_bottom_data_1_(new Blob<Dtype>()),
        blob_bottom_data_2_(new Blob<Dtype>()),
        blob_top_data_1_(new Blob<Dtype>()),
        blob_top_data_2_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_data_1_);
    blob_bottom_vec_.push_back(blob_bottom_data_2_);
    blob_top_vec_.push_back(blob_top_data_1_);
    blob_top_vec_.push_back(blob_top_data_2_);
  }

  virtual void Fill1ChannelData() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    // use 2 bottom blobs to do the test
    this->blob_bottom_data_1_->Reshape(2, 1, 6, 4);
    filler.Fill(this->blob_bottom_data_1_);
    this->blob_bottom_data_2_->Reshape(3, 1, 8, 5);
    filler.Fill(this->blob_bottom_data_2_);
  }

  virtual void Fill3ChannelData() {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    // use 2 bottom blobs to do the test
    this->blob_bottom_data_1_->Reshape(2, 3, 6, 4);
    filler.Fill(this->blob_bottom_data_1_);
    this->blob_bottom_data_2_->Reshape(3, 2, 8, 5);
    filler.Fill(this->blob_bottom_data_2_);
  }

  virtual ~TransformColorLayerTest() {
    delete blob_bottom_data_1_;
    delete blob_bottom_data_2_;
    delete blob_top_data_1_;
    delete blob_top_data_2_;
  }

  // transform 2 bottom blobs to do test
  Blob<Dtype>* const blob_bottom_data_1_;
  Blob<Dtype>* const blob_bottom_data_2_;
  Blob<Dtype>* const blob_top_data_1_;
  Blob<Dtype>* const blob_top_data_2_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TransformColorLayerTest, TestDtypesAndDevices);

TYPED_TEST(TransformColorLayerTest, TestChannel1to3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new TransformColorLayer<Dtype>(layer_param));
  this->Fill1ChannelData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call forward and check result
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_1_->num(), 2);
  EXPECT_EQ(this->blob_top_data_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_1_->height(), 6);
  EXPECT_EQ(this->blob_top_data_1_->width(), 4);
  // Test Data
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 6; ++h) {
        for (int w = 0; w < 4; ++w) {
          EXPECT_EQ(this->blob_top_data_1_->data_at(n, c, h, w),
            this->blob_bottom_data_1_->data_at(n, 0, h, w));
        }
      }
    }
  }

  EXPECT_EQ(this->blob_top_data_2_->num(), 3);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_2_->height(), 8);
  EXPECT_EQ(this->blob_top_data_2_->width(), 5);
  // Test Data
  for (int n = 0; n < 3; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 5; ++w) {
          EXPECT_EQ(this->blob_top_data_2_->data_at(n, c, h, w),
            this->blob_bottom_data_2_->data_at(n, 0, h, w));
        }
      }
    }
  }
}

TYPED_TEST(TransformColorLayerTest, TestChannel3to1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new TransformColorLayer<Dtype>(layer_param));
  this->Fill3ChannelData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call forward and check result
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_1_->num(), 2);
  EXPECT_EQ(this->blob_top_data_1_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_1_->height(), 6);
  EXPECT_EQ(this->blob_top_data_1_->width(), 4);
  // Test Data
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 1; ++c) {
      for (int h = 0; h < 6; ++h) {
        for (int w = 0; w < 4; ++w) {
          Dtype avg = (this->blob_bottom_data_1_->data_at(n, 0, h, w) +
          this->blob_bottom_data_1_->data_at(n, 1, h, w) +
          this->blob_bottom_data_1_->data_at(n, 2, h, w)) / 3;
          EXPECT_EQ(this->blob_top_data_1_->data_at(n, c, h, w), avg);
        }
      }
    }
  }

  EXPECT_EQ(this->blob_top_data_2_->num(), 3);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 8);
  EXPECT_EQ(this->blob_top_data_2_->width(), 5);
  // Test Data
  for (int n = 0; n < 3; ++n) {
    for (int c = 0; c < 1; ++c) {
      for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 5; ++w) {
          Dtype avg = (this->blob_bottom_data_2_->data_at(n, 0, h, w) +
          this->blob_bottom_data_2_->data_at(n, 1, h, w)) / 2;
          EXPECT_EQ(this->blob_top_data_2_->data_at(n, c, h, w), avg);
        }
      }
    }
  }
}

}  // namespace caffe
