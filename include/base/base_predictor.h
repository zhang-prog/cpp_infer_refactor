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

#ifndef BASE_PREDICTOR_H_
#define BASE_PREDICTOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/statusor.h"
#include "include/common/static_infer.h"
#include "include/utils/pp_option.h"

class BasePredictor {
 public:
  BasePredictor(
      const std::string &model_dir, const std::string &device = "cpu",
      const bool enable_mkldnn = false, int batch_size = 1,
      const std::unordered_map<std::string, std::string> &config = {});

  std::vector<BaseCVResult> Predict(string input);
  absl::Status LoadConfig();
  std::unique_ptr<PaddleInfer> CreateStaticInfer();

  const PaddlePredictorOption &PPOption();
  std::string ModelName();
  std::string ConfigPath();

  void SetBatchSize(int batch_size);

  virtual std::unordered_map<std::string, std::string> Process() = 0;
  virtual std::unique_ptr<BaseBatchSampler> BuildBatchSampler() = 0;
  virtual std::unique_ptr<BaseCVResult> GetResultClass() = 0;

 protected:
  std::string model_dir_;
  std::unordered_map<std::string, std::string> config_;
  int batch_size_;
  std::unique_ptr<FuncRegister> func_register_;
  std::unique_ptr<BaseBatchSampler> batch_sampler_;
  std::unique_ptr<BaseCVResult> result_class_;
  std::unique_ptr<PaddlePredictorOption> pp_option_;
};

#endif  // BASE_PREDICTOR_H_
