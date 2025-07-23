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

#ifndef TEXT_DET_RESULT_H
#define TEXT_DET_RESULT_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "include/base/base_cv_result.h"

class TextDetResult : public BaseCVResult {
 public:
  std::unordered_map<std::string, cv::Mat> ToImg() const override;
};

#endif  // TEXT_DET_RESULT_H
