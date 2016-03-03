#include <vector>

#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/utility_layers.hpp"

namespace caffe {

template <typename TypeParam>
class UnifiedLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnifiedLayerTest()
      : blob_bottom_data_1_(new Blob<Dtype>()),
        blob_bottom_data_2_(new Blob<Dtype>()),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_index_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_data_1_);
    blob_bottom_vec_.push_back(blob_bottom_data_2_);
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_index_);
  }

  virtual void FillBottomData() {
    // fill bottom blobs to test Forward()
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    // use 2 child layers to do test, one batch_size is 2, another is 3
    this->blob_bottom_data_1_->Reshape(2, 3, 6, 4);
    filler.Fill(this->blob_bottom_data_1_);
    this->blob_bottom_data_2_->Reshape(3, 3, 6, 4);
    filler.Fill(this->blob_bottom_data_2_);
  }

  virtual void FillTopData() {
    // fill top blobs to test Backward()
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    this->blob_top_data_->Reshape(5, 3, 6, 4);
    filler.Fill(this->blob_top_data_);
    // backward() only back-propagate diff
    // so copy data to diff, otherwise diff will be 0
    caffe_copy(this->blob_top_data_->count(),
      this->blob_top_data_->cpu_data(),
      static_cast<Dtype*>(this->blob_top_data_->mutable_cpu_diff()));
  }

  virtual ~UnifiedLayerTest() {
    delete blob_bottom_data_1_;
    delete blob_bottom_data_2_;
    delete blob_top_data_;
    delete blob_top_label_index_;
  }

  // merge 2 child layers into 1 unified_layer
  Blob<Dtype>* const blob_bottom_data_1_;
  Blob<Dtype>* const blob_bottom_data_2_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_index_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UnifiedLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnifiedLayerTest, TestLayerSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new UnifiedLayer<Dtype>(layer_param));
  this->FillBottomData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 6);
  EXPECT_EQ(this->blob_top_data_->width(), 4);

  EXPECT_EQ(this->blob_top_label_index_->num(), 2);
  EXPECT_EQ(this->blob_top_label_index_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_index_->height(), 1);
  EXPECT_EQ(this->blob_top_label_index_->width(), 1);
}

TYPED_TEST(UnifiedLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new UnifiedLayer<Dtype>(layer_param));
  this->FillBottomData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call forward and check result
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Test Data
  for (int n = 0; n < this->blob_top_data_->num(); ++n) {
    for (int c = 0; c < this->blob_top_data_->channels(); ++c) {
      for (int h = 0; h < this->blob_top_data_->height(); ++h) {
        for (int w = 0; w < this->blob_top_data_->width(); ++w) {
          EXPECT_EQ(this->blob_top_data_->data_at(n, c, h, w), n < 2 ?
            this->blob_bottom_data_1_->data_at(n, c, h, w)
            : this->blob_bottom_data_2_->data_at(n - 2, c, h, w));
        }
      }
    }
  }
  // Test Label_index
  EXPECT_EQ(this->blob_top_label_index_->data_at(0, 0, 0, 0), 2);
  EXPECT_EQ(this->blob_top_label_index_->data_at(1, 0, 0, 0), 3);
}

TYPED_TEST(UnifiedLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
      new UnifiedLayer<Dtype>(layer_param));
  // call fillBottomData first to shape the bottom blob
  this->FillBottomData();
  this->FillTopData();
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // Call backward and check result
  // use fake propagate_down only, for it's useless in DispatchLayer->Backward
  layer->Backward(this->blob_top_vec_, *new vector<bool>(),
    this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_data_1_->num(), 2);
  EXPECT_EQ(this->blob_bottom_data_1_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_data_1_->height(), 6);
  EXPECT_EQ(this->blob_bottom_data_1_->width(), 4);
  EXPECT_EQ(this->blob_bottom_data_2_->num(), 3);
  EXPECT_EQ(this->blob_bottom_data_2_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_data_2_->height(), 6);
  EXPECT_EQ(this->blob_bottom_data_2_->width(), 4);

  // Test Diff
  for (int n = 0; n < 5; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 6; ++h) {
        for (int w = 0; w < 4; ++w) {
          if (n < 2) {
            EXPECT_EQ(static_cast<Dtype>(5) / 2 * this->blob_top_data_->diff_at
              (n, c, h, w), this->blob_bottom_data_1_->diff_at(n, c, h, w));
          } else {
            EXPECT_EQ(static_cast<Dtype>(5) / 3 * this->blob_top_data_->diff_at
              (n, c, h, w), this->blob_bottom_data_2_->diff_at(n - 2, c, h, w));
          }
        }
      }
    }
  }
}

}  // namespace caffe
