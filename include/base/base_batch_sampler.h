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

#ifndef BASE_BATCH_SAMPLER_H
#define BASE_BATCH_SAMPLER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "absl/status/statusor.h"

class BaseBatchSampler {
 public:
  explicit BaseBatchSampler(int batch_size);

  int BatchSize() const;
  absl::Status SetBatchSize(int batch_size);
  absl::StatusOr<std::vector<cv::Mat>> Apply(const std::string& input);

  virtual absl::StatusOr<std::vector<cv::Mat>> SampleFromString(
      const std::string& input) = 0;
  virtual absl::StatusOr<std::vector<cv::Mat>> SampleFromVector(
      const std::vector<std::string>& inputs) = 0;

  template <typename T>
  absl::StatusOr<std::vector<cv::Mat>> Sample(const T& input);

 private:
  int batch_size_ = 1;
};

template <typename T>
absl::StatusOr<std::vector<cv::Mat>> BaseSampler::Sample(const T& input) {
  if (std::is_same<T, std::string>::value) {
    return SampleFromString(input);
  } else if (std::is_same<T, std::vector<std::string>>::value) {
    return SampleFromVector(input);
  } else {
    return absl::InvalidArgumentError(
        "Sample failed! Unsupported type for Sample");
  }
}

#endif  // BASE_BATCH_SAMPLER_H
