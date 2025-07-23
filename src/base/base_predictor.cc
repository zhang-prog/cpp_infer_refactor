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

#include "include/base/base_predictor.h"

#include <iostream>

#include "absl/status/statusor.h"
#include "include/base/base_batch_sampler.h"
#include "include/utils/pp_option.h"

BasePredictor::BasePredictor(
    const std::string &model_dir, const std::string &device,
    const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string> &config)
    : model_dir_(model_dir), batch_size_(batch_size), config_(config) {
  if (config.empty()) {
    config_ = LoadConfig();
  }
  pp_option_.reset(new PaddlePredictorOption());

  size_t pos = device.find(':');
  std::string device_type;
  int device_id;
  if (pos != std::string::npos) {
    device_type = device.substr(0, pos);
    device_id = std::stoi(device.substr(pos + 1));
  } else {
    device_type = device;
    device_id = 0;
  }
  auto result = pp_option_->SetDeviceType(device_type);
  if (!result.ok()) {
    std::cerr << result.ToString() << std::endl;
    return;
  }
  result = pp_option_->SetDeviceId(device_id);
  if (!result.ok()) {
    std::cerr << "Failed to set device id: " << result.ToString() << std::endl;
    return;
  }
  if (enable_mkldnn) {
    result = pp_option_->SetRunMode("mkldnn");
    if (!result.ok()) {
      std::cerr << "Failed to set run mode: " << result.ToString() << std::endl;
      return;
    }
  }
  std::cout << pp_option_->DebugString();
}

std::vector<BaseCVResult> BasePredictor::Predict(string input) {
    auto batches = batch_sampler_.apply(input)

        batches = self.batch_sampler(input)
        for batch_data in batches:
            prediction = self.process(batch_data, **kwargs)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))


    return {};
}

absl::Status BasePredictor::LoadConfig() { config_ = Yaml.load return {}; }

const PaddlePredictorOption &BasePredictor::PPOption() { return *pp_option_; }

std::string BasePredictor::ModelName() {
  return config_["Global"]["model_name"];
}

std::string BasePredictor::ConfigPath() { return ""; }

void BasePredictor::SetBatchSize(int batch_size) {}

std::unique_ptr<PaddleInfer> BasePredictor::CreateStaticInfer() {
  return nullptr;
}
