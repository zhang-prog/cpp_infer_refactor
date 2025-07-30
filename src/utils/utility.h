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

#ifndef UTILITY_H
#define UTILITY_H

#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#ifdef _WIN32
#include <direct.h>
#define mkdir _mkdir
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <errno.h>

class Utility {
 public:
  static constexpr const char* MODEL_FILE_PREFIX = "inference";

  static absl::StatusOr<
      std::map<std::string, std::pair<std::string, std::string>>>
  GetModelPaths(const std::string& model_dir,
                const std::string& model_file_prefix = MODEL_FILE_PREFIX);

  static absl::StatusOr<std::string> GetConfigPaths(
      const std::string& model_dir,
      const std::string& model_file_prefix = MODEL_FILE_PREFIX);
  static absl::Status FileExists(const std::string& path);

  // TODO windows
  static std::string GetCpuVendor();

  static void WriteBatchMatToTxt(const cv::Mat& batch,
                                 const std::string& filename);
  static void WriteBatchMatToTxt_X(const cv::Mat& mat,
                                   const std::string& filename);
  static void PrintShape(const cv::Mat& img);

  static absl::Status CreateDirectory(const std::string& path);
  static absl::Status CreatePath(const std::string& path);
  static absl::Status CreateFile(const std::string& filepath);
};

#endif  // UTILITY_H
