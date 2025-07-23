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

#include "utility.h"

#include <filesystem>

#include "absl/status/statusor.h"

namespace fs = std::filesystem;

absl::StatusOr<std::map<std::string, std::pair<std::string, std::string>>>
Utility::get_model_paths(const fs::path &model_dir,
                         const std::string &model_file_prefix) {
  std::map<std::string, std::vector<fs::path>> model_paths;
  fs::path pd_model_path;

  if (fs::exists(model_dir / (model_file_prefix + ".json"))) {
    pd_model_path = model_dir / (model_file_prefix + ".json");
  } else if (fs::exists(model_dir / (model_file_prefix + ".pdmodel"))) {
    pd_model_path = model_dir / (model_file_prefix + ".pdmodel");
  }
  if (pd_model_path.empty()) {
    return absl::NotFoundError(
        "No PaddlePaddle model file (.json or .pd) found!");
  }
  if (fs::exists(model_dir / (model_file_prefix + ".pdiparams"))) {
    model_paths["paddle"] = std::make_pair(
        pd_model_path.string(),
        (model_dir / (model_file_prefix + ".pdiparams")).string(), );
  } else {
    return absl::NotFoundError(
        "No PaddlePaddle params file (.pdiparams) found!");
  }

  return model_paths;
}
