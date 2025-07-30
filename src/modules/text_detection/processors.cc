// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "processors.h"

#include <stdexcept>

#include "src/utils/utility.h"

ReadImage::ReadImage(const std::string& format) {
  auto fmt = StringToFormat(format);
  if (!fmt.ok()) {
    std::cerr << fmt.status().ToString();
  }
  format_ = *fmt;
}

absl::StatusOr<std::vector<cv::Mat>> ReadImage::Apply(
    std::vector<cv::Mat>& input, const void* param_ptr) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  std::vector<cv::Mat> output;
  output.reserve(input.size());

  for (size_t i = 0; i < input.size(); ++i) {
    const cv::Mat& img = input[i];
    if (img.empty()) {
      return absl::InvalidArgumentError("Image at index " + std::to_string(i) +
                                        " is empty.");
    }

    cv::Mat converted;
    switch (format_) {
      case Format::BGR:
        if (img.channels() == 3) {
          converted = img.clone();
        } else if (img.channels() == 1) {
          cv::cvtColor(img, converted, cv::COLOR_GRAY2BGR);
        } else {
          return absl::InvalidArgumentError("Image at index " +
                                            std::to_string(i) +
                                            " channel not supported for BGR.");
        }
        break;
      case Format::RGB:
        if (img.channels() == 3) {
          cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
        } else if (img.channels() == 1) {
          cv::cvtColor(img, converted, cv::COLOR_GRAY2RGB);
        } else {
          return absl::InvalidArgumentError("Image at index " +
                                            std::to_string(i) +
                                            " channel not supported for RGB.");
        }
        break;
      case Format::GRAY:
        if (img.channels() == 3) {
          cv::cvtColor(img, converted, cv::COLOR_BGR2GRAY);
        } else if (img.channels() == 1) {
          converted = img.clone();
        } else {
          return absl::InvalidArgumentError("Image at index " +
                                            std::to_string(i) +
                                            " channel not supported for GRAY.");
        }
        break;
      default:
        return absl::InvalidArgumentError("Unknown format.");
    }
    output.push_back(std::move(converted));
  }
  return output;
}

absl::StatusOr<ReadImage::Format> ReadImage::StringToFormat(
    const std::string& format) {
  if (format == "BGR") return Format::BGR;
  if (format == "RGB") return Format::RGB;
  if (format == "GRAY") return Format::GRAY;
  return absl::InvalidArgumentError("Unsupported format: " + format);
}

DetResizeForTest::DetResizeForTest(int resize_long,
                                   std::vector<int> input_shape,
                                   std::vector<int> image_shape,
                                   int limit_side_len, std::string limit_type,
                                   int max_side_limit)
    : resize_long_(resize_long),
      input_shape_(input_shape),
      image_shape_(image_shape),
      limit_side_len_(limit_side_len),
      limit_type_(limit_type),
      max_side_limit_(max_side_limit) {
  resize_type_ = 0;
  if (!input_shape_.empty()) {
    resize_type_ = 3;
  } else if (!image_shape_.empty()) {
    resize_type_ = 1;
  } else if (limit_side_len_ > 0) {
    resize_type_ = 0;
  } else if (resize_long_ > 0) {
    resize_type_ = 2;
  }
};

absl::StatusOr<std::vector<cv::Mat>> DetResizeForTest::Apply(
    std::vector<cv::Mat>& input, const void* param_ptr) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  std::vector<cv::Mat> results;
  if (param_ptr != nullptr) {
    const DetResizeForTestParam* param =
        static_cast<const DetResizeForTestParam*>(param_ptr);
    for (const auto& img : input) {
      auto res = Resize(
          img,
          param->limit_side_len > 0 ? param->limit_side_len : limit_side_len_,
          !param->limit_type.empty() ? param->limit_type : limit_type_,
          param->max_side_limit > 0 ? param->max_side_limit : max_side_limit_);
      if (!res.ok()) return res.status();
      results.push_back(res.value());
    }
  } else {
    for (const auto& img : input) {
      auto res = Resize(img, limit_side_len_, limit_type_, max_side_limit_);
      if (!res.ok()) return res.status();
      results.push_back(res.value());
    }
  }
  return results;
}

absl::StatusOr<cv::Mat> DetResizeForTest::Resize(const cv::Mat& img,
                                                 int limit_side_len,
                                                 const std::string& limit_type,
                                                 int max_side_limit) const {
  int src_h = img.rows;
  int src_w = img.cols;
  if (src_h + src_w < 64) {
    cv::Mat padded = ImagePadding(img);
    src_h = padded.rows;
    src_w = padded.cols;
    return Resize(padded, limit_side_len, limit_type, max_side_limit);
  }

  switch (resize_type_) {
    case 0:
      return ResizeImageType0(img, limit_side_len, limit_type, max_side_limit);
    case 1:
      return ResizeImageType1(img);
    case 2:
      return ResizeImageType2(img);
    case 3:
      return ResizeImageType3(img);
    default:
      return absl::InvalidArgumentError("Unknown resize_type: " +
                                        std::to_string(resize_type_));
  }
}

cv::Mat DetResizeForTest::ImagePadding(const cv::Mat& img, int value) const {
  int h = img.rows, w = img.cols, c = img.channels();
  int pad_h = std::max(32, h);
  int pad_w = std::max(32, w);
  cv::Mat im_pad = cv::Mat::zeros(pad_h, pad_w, img.type());
  im_pad.setTo(cv::Scalar::all(value));
  img.copyTo(im_pad(cv::Rect(0, 0, w, h)));
  return im_pad;
}

absl::StatusOr<cv::Mat> DetResizeForTest::ResizeImageType0(
    const cv::Mat& img, int limit_side_len, const std::string& limit_type,
    int max_side_limit) const {
  int h = img.rows, w = img.cols;
  float ratio = 1.f;
  if (limit_type == "max") {
    if (std::max(h, w) > limit_side_len)
      ratio = float(limit_side_len) / std::max(h, w);
  } else if (limit_type == "min") {
    if (std::min(h, w) < limit_side_len)
      ratio = float(limit_side_len) / std::min(h, w);
  } else if (limit_type == "resize_long") {
    ratio = float(limit_side_len) / std::max(h, w);
  } else {
    return absl::InvalidArgumentError("Not supported limit_type: " +
                                      limit_type);
  }
  int resize_h = int(h * ratio);
  int resize_w = int(w * ratio);

  if (std::max(resize_h, resize_w) > max_side_limit) {
    ratio = float(max_side_limit) / std::max(resize_h, resize_w);
    resize_h = int(resize_h * ratio);
    resize_w = int(resize_w * ratio);
  }
  resize_h = std::max(int(std::round(resize_h / 32.0) * 32), 32);
  resize_w = std::max(int(std::round(resize_w / 32.0) * 32), 32);

  if (resize_h == h && resize_w == w) return img;
  if (resize_h <= 0 || resize_w <= 0)
    return absl::InvalidArgumentError("resize_w/h <= 0");
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat> DetResizeForTest::ResizeImageType1(
    const cv::Mat& img) const {
  int resize_h = image_shape_[0];
  int resize_w = image_shape_[1];
  int ori_h = img.rows, ori_w = img.cols;
  if (keep_ratio_) {
    resize_w = int(ori_w * (float(resize_h) / ori_h));
    int N = int(std::ceil(resize_w / 32.0));
    resize_w = N * 32;
  }
  if (resize_h == ori_h && resize_w == ori_w) return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat> DetResizeForTest::ResizeImageType2(
    const cv::Mat& img) const {
  int h = img.rows, w = img.cols;
  int resize_h = h, resize_w = w;
  float ratio;
  if (resize_h > resize_w)
    ratio = float(resize_long_) / resize_h;
  else
    ratio = float(resize_long_) / resize_w;

  resize_h = int(resize_h * ratio);
  resize_w = int(resize_w * ratio);

  int max_stride = 128;
  resize_h = ((resize_h + max_stride - 1) / max_stride) * max_stride;
  resize_w = ((resize_w + max_stride - 1) / max_stride) * max_stride;

  if (resize_h == h && resize_w == w) return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

absl::StatusOr<cv::Mat> DetResizeForTest::ResizeImageType3(
    const cv::Mat& img) const {
  if (input_shape_.size() != INPUTSHAPE)
    return absl::InvalidArgumentError("input_shape not set for type " +
                                      std::to_string(INPUTSHAPE));
  int resize_h = input_shape_[1];
  int resize_w = input_shape_[2];
  int ori_h = img.rows, ori_w = img.cols;
  if (resize_h == ori_h && resize_w == ori_w) return img;
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));
  return resized;
}

NormalizeImage::NormalizeImage(double scale, const std::vector<double>& mean,
                               const std::vector<double>& std)
    : alpha_(CHANNEL), beta_(CHANNEL) {
  for (size_t i = 0; i < CHANNEL; ++i) {
    alpha_[i] = scale / std.at(i);
    beta_[i] = -mean.at(i) / std.at(i);
  }
}

absl::StatusOr<cv::Mat> NormalizeImage::Normalize(const cv::Mat& img) const {
  if (img.empty()) {
    return absl::InvalidArgumentError("Input image is empty.");
  }
  if (img.channels() != CHANNEL) {
    return absl::InvalidArgumentError("Input image must have 3 channels.");
  }
  if (img.depth() != CV_8U && img.depth() != CV_32F) {
    return absl::InvalidArgumentError("Input image must be CV_8U or CV_32F.");
  }

  cv::Mat input;
  // 转 float
  if (img.depth() == CV_8U) {
    img.convertTo(input, CV_32F);
  } else {
    input = img.clone();
  }

  cv::Mat processed = input;

  std::vector<cv::Mat> channels(CHANNEL);
  cv::split(processed, channels);

  for (int c = 0; c < CHANNEL; ++c) {
    channels[c] = channels[c] * alpha_[c] + beta_[c];
  }

  cv::merge(channels, processed);
  return processed;
}

absl::StatusOr<std::vector<cv::Mat>> NormalizeImage::Apply(
    std::vector<cv::Mat>& imgs, const void* param) const {
  std::vector<cv::Mat> results;
  results.reserve(imgs.size());
  for (const auto& img : imgs) {
    auto normed = this->Normalize(img);
    if (!normed.ok()) {
      return normed.status();
    }
    results.push_back(std::move(normed).value());
  }
  return results;
}

absl::StatusOr<std::vector<cv::Mat>> ToCHWImage::operator()(
    const std::vector<cv::Mat>& imgs_batch) {
  std::vector<std::vector<cv::Mat>> chw_imgs_batch;

  std::vector<cv::Mat> chw_imgs;
  for (const auto& img : imgs_batch) {
    if (img.empty()) {
      return absl::InvalidArgumentError("Input image is empty!");
    }
    if (img.channels() != 3) {
      return absl::InvalidArgumentError(
          "Input image must have 3 channels (HWC format)!");
    }

    cv::Mat chw_img(3, img.rows * img.cols, CV_32F);
    float* ptr = chw_img.ptr<float>();

    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
        const cv::Vec3b& pixel = img.at<cv::Vec3b>(h, w);
        ptr[0 * img.total() + h * img.cols + w] = pixel[0];
        ptr[1 * img.total() + h * img.cols + w] = pixel[1];
        ptr[2 * img.total() + h * img.cols + w] = pixel[2];
      }
    }

    chw_imgs.push_back(chw_img);
  }

  return chw_imgs;
}

absl::StatusOr<std::vector<cv::Mat>> ToCHWImage::Apply(
    std::vector<cv::Mat>& input, const void* param) const {
  std::vector<cv::Mat> chw_imgs;
  for (const auto& img : input) {
    if (img.empty()) {
      return absl::InvalidArgumentError("Input image is empty!");
    }
    if (img.channels() != 3) {
      return absl::InvalidArgumentError(
          "Input image must have 3 channels (HWC format)!");
    }

    std::vector<int> sizes = {3, img.rows, img.cols};  // Define sizes for CHW
    cv::Mat chw_img(3, sizes.data(), CV_32F);
    float* ptr = chw_img.ptr<float>();
    for (int h = 0; h < img.rows; ++h) {
      for (int w = 0; w < img.cols; ++w) {
        const cv::Vec3f& pixel = img.at<cv::Vec3f>(h, w);
        ptr[0 * img.total() + h * img.cols + w] = pixel[0];
        ptr[1 * img.total() + h * img.cols + w] = pixel[1];
        ptr[2 * img.total() + h * img.cols + w] = pixel[2];
      }
    }

    chw_imgs.push_back(chw_img);
  }

  return chw_imgs;
}

absl::StatusOr<std::vector<cv::Mat>> ToBatch::operator()(
    const std::vector<cv::Mat>& imgs) const {
  if (imgs.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  const int batch = imgs.size();
  const int rows = imgs[0].rows;
  const int cols = imgs[0].cols;
  const int channels = imgs[0].channels();

  for (size_t i = 0; i < imgs.size(); ++i) {
    if (imgs[i].rows != rows || imgs[i].cols != cols ||
        imgs[i].channels() != channels) {
      return absl::InvalidArgumentError(
          "All images must have the same size and number of channels.");
    }
  }

  std::vector<int> sizes = {batch, rows, cols, channels};
  cv::Mat out(4, sizes.data(), CV_32F);

  for (int b = 0; b < batch; ++b) {
    cv::Mat img_float;
    if (imgs[b].depth() != CV_32F) {
      imgs[b].convertTo(img_float, CV_32F);
    } else {
      img_float = imgs[b];
    }

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (channels == 1) {
          float v = img_float.at<float>(r, c);
          int idx[4] = {b, r, c, 0};
          out.at<float>(idx) = v;
        } else if (channels == 3) {
          cv::Vec3f v = img_float.at<cv::Vec3f>(r, c);
          for (int ch = 0; ch < 3; ++ch) {
            int idx[4] = {b, r, c, ch};
            out.at<float>(idx) = v[ch];
          }
        } else {
          const float* pix = img_float.ptr<float>(r, c);
          for (int ch = 0; ch < channels; ++ch) {
            int idx[4] = {b, r, c, ch};
            out.at<float>(idx) = pix[ch];
          }
        }
      }
    }
  }
  std::vector<cv::Mat> result{out};
  return result;
}

absl::StatusOr<std::vector<cv::Mat>> ToBatch::Apply(std::vector<cv::Mat>& input,
                                                    const void* param) const {
  if (input.empty()) {
    return absl::InvalidArgumentError("Input image vector is empty.");
  }
  const int batch = input.size();
  const int rows = input[0].size[1];
  const int cols = input[0].size[2];
  const int channels = input[0].size[0];

  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i].size[1] != rows || input[i].size[2] != cols ||
        input[i].size[0] != channels) {
      return absl::InvalidArgumentError(
          "All images must have the same size and number of channels.");
    }
  }

  std::vector<int> sizes = {batch, channels, rows, cols};
  cv::Mat out(4, sizes.data(), CV_32F);

  for (int b = 0; b < batch; ++b) {
    cv::Mat img_float;
    if (input[b].depth() != CV_32F) {
      input[b].convertTo(img_float, CV_32F);
    } else {
      img_float = input[b];
    }

    for (int ch = 0; ch < channels; ++ch) {
      for (int r = 0; r < rows; ++r) {
        const float* row_ptr = img_float.ptr<float>(ch, r);
        for (int c = 0; c < cols; ++c) {
          int idx[4] = {b, ch, r, c};
          out.at<float>(idx) = row_ptr[c];
        }
      }
    }
  }
  std::vector<cv::Mat> result{out};
  return result;
}

DBPostProcess::DBPostProcess(float thresh, float box_thresh, int max_candidates,
                             float unclip_ratio, bool use_dilation,
                             const std::string& score_mode,
                             const std::string& box_type)
    : thresh_(thresh),
      box_thresh_(box_thresh),
      max_candidates_(max_candidates),
      unclip_ratio_(unclip_ratio),
      min_size_(3),
      use_dilation_(use_dilation),
      score_mode_(score_mode),
      box_type_(box_type) {
  assert(score_mode == "slow" || score_mode == "fast");
  assert(box_type == "quad" || box_type == "poly");
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::operator()(const cv::Mat& preds,
                          const std::vector<int>& img_shapes,
                          absl::optional<float> thresh,
                          absl::optional<float> box_thresh,
                          absl::optional<float> unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> all_boxes;
  std::vector<float> all_scores;
  auto preds_batch = SplitBatch(preds);
  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  for (const auto& preds_data : *preds_batch) {
    auto result = Process(preds_data, img_shapes, thresh.value_or(thresh_),
                          box_thresh.value_or(box_thresh_),
                          unclip_ratio.value_or(unclip_ratio_));

    if (!result.ok()) {
      return result.status();
    }

    auto boxes_result = *result;
    auto boxes = boxes_result.first;
    auto scores = boxes_result.second;
    all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    all_scores.insert(all_scores.end(), scores.begin(), scores.end());
  }

  return std::make_pair(all_boxes, all_scores);
}

absl::StatusOr<std::vector<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>>
DBPostProcess::Apply(const cv::Mat& preds, const std::vector<int>& img_shapes,
                     absl::optional<float> thresh,
                     absl::optional<float> box_thresh,
                     absl::optional<float> unclip_ratio) {
  std::vector<
      std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
      db_result = {};
  std::vector<std::vector<cv::Point2f>> all_boxes = {};
  std::vector<float> all_scores = {};

  auto preds_batch = SplitBatch(preds);

  if (!preds_batch.ok()) {
    return preds_batch.status();
  }
  for (const auto& preds_data : *preds_batch) {
    auto result = Process(preds, img_shapes, thresh.value_or(thresh_),
                          box_thresh.value_or(box_thresh_),
                          unclip_ratio.value_or(unclip_ratio_));

    if (!result.ok()) {
      return result.status();
    }

    auto boxes_result = *result;
    auto boxes = boxes_result.first;
    auto scores = boxes_result.second;
    all_boxes.insert(all_boxes.end(), boxes.begin(), boxes.end());
    all_scores.insert(all_scores.end(), scores.begin(), scores.end());
    db_result.push_back(std::make_pair(all_boxes, all_scores));
    all_boxes.clear();
    all_scores.clear();
  }

  return db_result;
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::Process(const cv::Mat& pred, const std::vector<int>& img_shape,
                       float thresh, float box_thresh, float unclip_ratio) {
  cv::Mat pred_single = pred.clone();
  cv::Mat result = pred_single.reshape(1, pred_single.size[2]);
  cv::Mat segmentation = result > thresh;
  cv::Mat mask;
  if (use_dilation_) {
    cv::Mat kernel = (cv::Mat_<uchar>(2, 2) << 1, 1, 1, 1);  //暂时未测试
    cv::dilate(segmentation, mask, kernel);
  } else {
    mask = segmentation;
  }

  int src_h = img_shape[0];
  int src_w = img_shape[1];

  if (box_type_ == "poly") {
    return PolygonsFromBitmap(pred_single, mask, src_w, src_h, box_thresh,
                              unclip_ratio);
  } else if (box_type_ == "quad") {
    return BoxesFromBitmap(pred_single, mask, src_w, src_h, box_thresh,
                           unclip_ratio);
  }

  return absl::InvalidArgumentError(
      "box_type can only be one of ['quad', 'poly']");
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::PolygonsFromBitmap(const cv::Mat& pred, const cv::Mat& bitmap,
                                  int dest_width, int dest_height,
                                  float box_thresh, float unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> boxes;
  std::vector<float> scores;

  float width_scale = static_cast<float>(dest_width) / bitmap.cols;
  float height_scale = static_cast<float>(dest_height) / bitmap.rows;

  cv::Mat bitmap_uint8;
  bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

  std::vector<std::vector<cv::Point2f>> contours;
  cv::findContours(bitmap_uint8, contours, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  int num_contours =
      std::min(static_cast<int>(contours.size()), max_candidates_);

  for (int i = 0; i < num_contours; ++i) {
    const auto& contour = contours[i];

    std::vector<cv::Point2f> approx;
    double epsilon = 0.002 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    if (approx.size() < 4) {
      continue;
    }

    float score = BoxScoreFast(pred, approx);
    if (box_thresh > score) {
      continue;
    }

    std::vector<cv::Point2f> box;
    if (approx.size() > 2) {
      auto unclip_result = Unclip(approx, unclip_ratio);
      if (!unclip_result.ok()) {
        continue;
      }
      box = *unclip_result;
      if (box.size() > 1) {
        continue;
      }
    } else {
      continue;
    }

    if (!box.empty()) {
      auto min_box_result = GetMiniBoxes(box);
      auto min_box = min_box_result.first;
      auto sside = min_box_result.second;
      if (sside < min_size_ + 2) {
        continue;
      }

      for (auto& point : box) {
        point.x = std::max(
            0, std::min(static_cast<int>(std::round(point.x * width_scale)),
                        dest_width - 1));
        point.y = std::max(
            0, std::min(static_cast<int>(std::round(point.y * height_scale)),
                        dest_height - 1));
      }

      boxes.push_back(box);
      scores.push_back(score);
    }
  }

  return std::make_pair(boxes, scores);
}

absl::StatusOr<
    std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
DBPostProcess::BoxesFromBitmap(const cv::Mat& pred, const cv::Mat& bitmap,
                               int dest_width, int dest_height,
                               float box_thresh, float unclip_ratio) {
  std::vector<std::vector<cv::Point2f>> boxes;
  std::vector<float> scores;

  float width_scale = static_cast<float>(dest_width) / bitmap.cols;
  float height_scale = static_cast<float>(dest_height) / bitmap.rows;

  cv::Mat bitmap_uint8;
  bitmap.convertTo(bitmap_uint8, CV_8UC1, 255.0);

  std::vector<std::vector<cv::Point>> contours_;
  cv::findContours(bitmap_uint8, contours_, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point2f>> contours;
  for (const auto& contour : contours_) {
    std::vector<cv::Point2f> float_contour;
    for (const auto& point : contour) {
      float_contour.push_back(cv::Point2f(point.x, point.y));
    }
    contours.push_back(float_contour);
  }
  int num_contours =
      std::min(static_cast<int>(contours.size()), max_candidates_);

  for (int i = 0; i < num_contours; ++i) {
    const auto& contour = contours[i];

    auto contour_result = GetMiniBoxes(contour);
    auto points = contour_result.first;
    auto sside = contour_result.second;
    if (sside < min_size_) {
      continue;
    }

    float score = 0;
    if (score_mode_ == "fast") {
      score = BoxScoreFast(pred, points);
    } else {
      score = BoxScoreSlow(pred, contour);
    }

    if (box_thresh > score) {
      continue;
    }

    auto unclip_result = Unclip(points, unclip_ratio);
    if (!unclip_result.ok()) {
      continue;
    }

    auto box = *unclip_result;
    auto min_box_result = GetMiniBoxes(box);
    auto min_box = min_box_result.first;
    auto new_sside = min_box_result.second;
    if (new_sside < min_size_ + 2) {
      continue;
    }

    for (auto& point : min_box) {
      point.x = std::max(
          0, std::min(static_cast<int>(std::round(point.x * width_scale)),
                      dest_width - 1));
      point.y = std::max(
          0, std::min(static_cast<int>(std::round(point.y * height_scale)),
                      dest_height - 1));
    }

    boxes.push_back(min_box);
    scores.push_back(score);
  }

  return std::make_pair(boxes, scores);
}

absl::StatusOr<std::vector<cv::Point2f>> DBPostProcess::Unclip(
    const std::vector<cv::Point2f>& box, float unclip_ratio) {
  float area = cv::contourArea(box);
  float length = cv::arcLength(box, true);
  float distance = area * unclip_ratio / length;

  ClipperLib::Path path;
  for (const auto& point : box) {
    path << ClipperLib::IntPoint(point.x, point.y);
  }

  ClipperLib::ClipperOffset co;
  co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths solution;
  co.Execute(solution, distance);

  if (solution.empty()) {
    return absl::InternalError("Failed to unclip polygon");
  }

  std::vector<cv::Point2f> result;
  for (const auto& p : solution[0]) {
    result.emplace_back(p.X, p.Y);
  }

  return result;
}

std::pair<std::vector<cv::Point2f>, float> DBPostProcess::GetMiniBoxes(
    const std::vector<cv::Point2f>& contour) {
  cv::RotatedRect box = cv::minAreaRect(contour);

  cv::Point2f vertex[4];
  box.points(vertex);

  std::vector<cv::Point2f> points(vertex, vertex + 4);
  std::sort(
      points.begin(), points.end(),
      [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

  int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
  if (points[1].y > points[0].y) {
    index_1 = 0;
    index_4 = 1;
  } else {
    index_1 = 1;
    index_4 = 0;
  }

  if (points[3].y > points[2].y) {
    index_2 = 2;
    index_3 = 3;
  } else {
    index_2 = 3;
    index_3 = 2;
  }

  std::vector<cv::Point2f> box_points = {points[index_1], points[index_2],
                                         points[index_3], points[index_4]};

  float sside = std::min(box.size.width, box.size.height);
  return std::make_pair(box_points, sside);
}

cv::Mat DBPostProcess::extract2DFromBitmap(const cv::Mat& bitmap, int ymin,
                                           int ymax, int xmin, int xmax,
                                           int batch_idx, int channel_idx) {
  int roi_height = ymax - ymin + 1;
  int roi_width = xmax - xmin + 1;

  cv::Mat roi(roi_height, roi_width, bitmap.type());
  for (int y = ymin; y <= ymax; y++) {
    for (int x = xmin; x <= xmax; x++) {
      int indices[] = {batch_idx, channel_idx, y, x};
      float value = bitmap.at<float>(indices);
      roi.at<float>(y - ymin, x - xmin) = value;
    }
  }

  return roi;
}

float DBPostProcess::BoxScoreFast(const cv::Mat& bitmap,
                                  const std::vector<cv::Point2f>& box) {
  int h, w;
  if (bitmap.dims == 4) {
    // 获取最后两个维度作为 height 和 width
    h = bitmap.size[bitmap.dims - 2];  // height
    w = bitmap.size[bitmap.dims - 1];  // width
  } else if (bitmap.dims == 2) {
    h = bitmap.rows;
    w = bitmap.cols;
  } else {
    throw std::runtime_error("Unsupported bitmap dimensions");
  }

  // Copy the box
  std::vector<cv::Point2f> box_copy = box;

  // Find min/max coordinates
  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::lowest();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::lowest();

  for (const auto& point : box) {
    x_min = std::min(x_min, point.x);
    x_max = std::max(x_max, point.x);
    y_min = std::min(y_min, point.y);
    y_max = std::max(y_max, point.y);
  }

  int xmin = std::max(0, std::min(static_cast<int>(std::floor(x_min)), w - 1));
  int xmax = std::max(0, std::min(static_cast<int>(std::ceil(x_max)), w - 1));
  int ymin = std::max(0, std::min(static_cast<int>(std::floor(y_min)), h - 1));
  int ymax = std::max(0, std::min(static_cast<int>(std::ceil(y_max)), h - 1));

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  std::vector<cv::Point> box_int;
  for (auto& point : box_copy) {
    point.x -= xmin;
    point.y -= ymin;
    box_int.push_back(
        cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
  }

  std::vector<std::vector<cv::Point>> contours = {box_int};
  cv::fillPoly(mask, contours, cv::Scalar(1));

  cv::Mat roi = extract2DFromBitmap(bitmap, ymin, ymax, xmin, xmax, 0, 0);

  cv::Scalar mean_val = cv::mean(roi, mask);

  return mean_val[0];
}

float DBPostProcess::BoxScoreSlow(const cv::Mat& bitmap,
                                  const std::vector<cv::Point2f>& contour) {
  int h = bitmap.rows;
  int w = bitmap.cols;

  std::vector<cv::Point2f> contour_copy = contour;

  int xmin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f& a, const cv::Point2f& b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int xmax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f& a, const cv::Point2f& b) {
                                return a.x < b.x;
                              })
                 ->x)));
  int ymin = std::max(
      0, static_cast<int>(std::floor(
             std::min_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f& a, const cv::Point2f& b) {
                                return a.y < b.y;
                              })
                 ->y)));
  int ymax = std::max(
      0, static_cast<int>(std::ceil(
             std::max_element(contour_copy.begin(), contour_copy.end(),
                              [](const cv::Point2f& a, const cv::Point2f& b) {
                                return a.y < b.y;
                              })
                 ->y)));

  xmin = std::min(xmin, w - 1);
  xmax = std::min(xmax, w - 1);
  ymin = std::min(ymin, h - 1);
  ymax = std::min(ymax, h - 1);

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  for (auto& point : contour_copy) {
    point.x -= xmin;
    point.y -= ymin;
  }

  std::vector<std::vector<cv::Point2f>> contours = {contour_copy};
  cv::fillPoly(mask, contours, 1);

  cv::Scalar mean = cv::mean(
      bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask);
  return static_cast<float>(mean[0]);
}

absl::StatusOr<std::vector<cv::Mat>> DBPostProcess::SplitBatch(
    const cv::Mat& batch) {
  if (batch.dims != 4) {
    return absl::InvalidArgumentError("Input batch must be a 4D cv::Mat.");
  }
  if (batch.type() != CV_32F && batch.type() != CV_32F) {
    return absl::InvalidArgumentError(
        "Input batch must have CV_32F element type.");
  }

  std::vector<cv::Mat> split_mats;
  for (int i = 0; i < batch.size[0]; i++) {
    cv::Range ranges[4];
    ranges[0] = cv::Range(i, i + 1);
    ranges[1] = cv::Range::all();
    ranges[2] = cv::Range::all();
    ranges[3] = cv::Range::all();

    cv::Mat sub_mat = batch(ranges);
    split_mats.push_back(sub_mat);
  }

  return split_mats;
}
