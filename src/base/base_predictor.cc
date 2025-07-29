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

#include "base_predictor.h"

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "base_batch_sampler.h"
#include "src/utils/pp_option.h"
#include "src/utils/utility.h"

BasePredictor::BasePredictor(
    const std::string& model_dir, const std::string& device,
    const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string>& config)
    : model_dir_(model_dir), batch_size_(batch_size), config_(config) {
  if (config.empty()) {
    auto result = LoadConfig("/workspace/cpp_infer_refactor/inference.yml");
    if (!result.ok()) {
      std::cerr << result.ToString();
    }
  }
  pp_option_ptr_.reset(new PaddlePredictorOption());

  size_t pos = device.find(':');
  std::string device_type = "";
  int device_id = 0;
  if (pos != std::string::npos) {
    device_type = device.substr(0, pos);
    device_id = std::stoi(device.substr(pos + 1));
  } else {
    device_type = device;
    device_id = 0;
  }
  auto result = pp_option_ptr_->SetDeviceType(device_type);
  if (!result.ok()) {
    std::cerr << result.ToString() << std::endl;
    return;
  }
  result = pp_option_ptr_->SetDeviceId(device_id);
  if (!result.ok()) {
    std::cerr << "Failed to set device id: " << result.ToString() << std::endl;
    return;
  }
  if (enable_mkldnn) {
    result = pp_option_ptr_->SetRunMode("mkldnn");
    if (!result.ok()) {
      std::cerr << "Failed to set run mode: " << result.ToString() << std::endl;
      return;
    }
  }
  std::cout << pp_option_ptr_->DebugString();
}

std::vector<std::unique_ptr<BaseCVResult>> BasePredictor::Predict(
    const std::string& input) {
  input_path_ = input;
  std::vector<std::unique_ptr<BaseCVResult>> result = {};
  auto batches = batch_sampler_ptr_->Apply(input);
  if (!batches.ok()) {
    std::cerr << "Get sample fail : " << batches.status().ToString();
  }
  for (auto& batch_data : batches.value()) {
    auto predictions = Process(batch_data);
    for (auto& prediction : predictions) {
      result.emplace_back(std::move(prediction));
    }
  }
  return result;
}

absl::Status BasePredictor::LoadConfig(const std::string& config_path) {
  config_ = YamlConfig(config_path);
  ;
  return absl::OkStatus();
}

const PaddlePredictorOption& BasePredictor::PPOption() {
  return *pp_option_ptr_;
}

absl::StatusOr<std::string> BasePredictor::GetModelName() {
  auto model_name = config_.GetString(std::string("Global.model_name"));
  if (!model_name.ok()) {
    return model_name.status();
  }
  return model_name;
}

std::string BasePredictor::ConfigPath() {
  auto config_path = Utility::GetConfigPaths(model_dir_);
  if (config_path.ok()) {
    return config_path.value();
  }
  return "";
}

void BasePredictor::SetBatchSize(int batch_size) { batch_size_ = batch_size; }

std::unique_ptr<PaddleInfer> BasePredictor::CreateStaticInfer() {
  batch_sampler_ptr_ = BuildBatchSampler();  //**********
  // result_class_ptr_ = GetResultClass();  //已在实现类直接返回 baseCVResult
  auto model_name = GetModelName();
  if (!model_name.ok()) {
    std::cerr << "Could find model:" << model_name.status().ToString();
  }
  return std::unique_ptr<PaddleInfer>(
      new PaddleInfer(*model_name, model_dir_, MODEL_FILE_PREFIX, PPOption()));
}
