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

#ifndef IMAGE_BATCH_SAMPLER_H_
#define IMAGE_BATCH_SAMPLER_H_

#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/base/base_batch_sampler.h"

class ImageBatchSampler : public BaseBatchSampler {
 public:
  explicit ImageBatchSampler(int batch_size = 1);
  virtual ~ImageBatchSampler() {}  //这里还没调研怎么实现  ？？？

  absl::StatusOr<std::vector<std::vector<cv::Mat> > > SampleFromString(
      const std::string& input) override;

  absl::StatusOr<std::vector<std::vector<cv::Mat> > > SampleFromVector(
      const std::vector<std::string>& inputs) override;

 private:
  absl::StatusOr<std::vector<std::string> > GetFilesList(
      const std::string& path);
  absl::StatusOr<cv::Mat> LoadImage(const std::string& file_path);
  bool IsImageFile(const std::string& file_path) const;
  bool IsDirectory(const std::string& path) const;
  bool FileExists(const std::string& path) const;
  std::string GetFileExtension(const std::string& file_path) const;
  void GetFilesRecursive(const std::string& dir_path,
                         std::vector<std::string>& file_list) const;
  std::string ToLower(const std::string& str) const;

  static const std::set<std::string> kImgSuffixes;
};

#endif  // IMAGE_BATCH_SAMPLER_H_
