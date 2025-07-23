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

#ifndef IMAGE_BATCH_SAMPLER_H
#define IMAGE_BATCH_SAMPLER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "absl/status/statusor.h"
#include "include/base/base_batch_sampler.h"

class ImageBatchSampler : public BaseBatchSampler {
 public:
  absl::StatusOr<std::vector<cv::Mat>> SampleFromString(
      const std::string& input) override;
  absl::StatusOr<std::vector<cv::Mat>> SampleFromVector(
      const std::vector<std::string>& inputs) override;
};

#endif  // IMAGE_BATCH_SAMPLER_H
