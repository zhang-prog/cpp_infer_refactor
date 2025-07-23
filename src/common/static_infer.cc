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

#include "include/common/static_infer.h"

#include "absl/status/statusor.h"

PaddleInfer::PaddleInfer(const std::string &model_name,
                         const std::string &model_dir,
                         const std::string &model_file_prefix,
                         const PaddlePredictorOption &option)
    : model_name_(model_name),
      model_dir_(model_dir),
      model_file_prefix_(model_file_prefix),
      option_(option) {
  auto result = Create();
  if (!result.ok()) {
    std::cerr << "Create predictor failed: " << result.status() << std::endl;
    return;
  }

  predictor_ = std::move(result.value());
  auto input_names = predictor_->GetInputNames();
  for (const auto &name : input_names) {
    auto handle = predictor_->GetInputHandle(name);
    input_handles_.emplace_back(std::move(handle));
  }
  auto output_names = predictor_->GetOutputNames();
  for (const auto &name : output_names) {
    auto handle = predictor_->GetOutputHandle(name);
    output_handles_.emplace_back(std::move(handle));
  }
}

absl::StatusOr<std::unique_ptr<paddle_infer::Predictor>> Create() {
  auto model_paths = get_model_paths(model_dir_, model_file_prefix_);
  if (!model_paths.ok()) {
    return model_paths.status();
  }
  if (model_paths.find("paddle") == model_paths.end()) {
    return absl::NotFoundError("No valid PaddlePaddle model found");
  }

  auto result = CheckRunMode();
  if (!result.ok()) {
    return result.status();
  }

  auto model_files = model_paths["paddle"];
  std::string model_file = model_files.first;
  std::string params_file = model_files.second;

  if (option_.device_type == "cpu" && option_.device_id != nullptr) {
    option_.device_id = nullptr;
    std::cout << "`device_id` has been set to nullptr" << std::endl;
  }

  if (option_.device_type == "gpu" && option_.device_id == nullptr) {
    option_.device_id = 0;
    std::cout << "`device_id` has been set to 0" << std::endl;
  }

  paddle_infer::Config config;
  config.SetModel(model_file, params_file);

  if (option_.device_type == "gpu") {
    std::unordered_set<std::string> mixed_op_set = {"feed", "fetch"};
    config.ExpDisableMixPrecisionOps(mixed_op_set);

    paddle_infer::PrecisionType precision =
        paddle_infer::PrecisionType::kFloat32;
    if (option_.run_mode == "paddle_fp16") {
      precision = paddle_infer::PrecisionType::kHalf;
    }

    config.DisableMKLDNN();
    config.EnableUseGpu(100, option_.device_id, precision);
    config.EnableNewIR(option_.enable_new_ir);
    if (option_.enable_new_ir && option_.enable_cinn) {
      config.EnableCINN();
    }
    config.EnableNewExecutor();
    config.SetOptimizationLevel(3);
  } else if (option_.device_type == "cpu") {
    config.DisableGpu();
    if (option_.run_mode.find("mkldnn") != std::string::npos) {
      config.EnableMkldnn();
      if (option_.run_mode.find("bf16") != std::string::npos) {
        config.EnableMkldnnBfloat16();
      }
      config.SetMkldnnCacheCapacity(option_.mkldnn_cache_capacity);
    } else {
      config.DisableMKLDNN();
    }
    config.SetCpuMathLibraryNumThreads(option_.cpu_threads);
    config.EnableNewIr(option_.enable_new_ir);
    config.EnableNewExecutor();
    config.SetOptimizationLevel(3);
  } else {
    return absl::InvalidArgumentError("Not supported device type: " +
                                      option_.device_type);
  }

  config.EnableMemoryOptim();
  for (const auto &del_p : option_.delete_pass) {
    config.DeletePass(del_p);
  }
  config.DisableGlogInfo();

  auto predictor = paddle_infer::CreatePredictor(config);

  return predictor;
}

absl::StatusOr < std::vector<std::vector<float>> PaddleInfer::Apply(
                     const std::vector<cv::Mat> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    auto &input_handle = input_handles_[i];
    input_handle->Reshape({static_cast<int>(x[i].size())});
    input_handle->CopyFromCpu(x[i].data());
  }
  predictor_->Run();

  std::vector<std::vector<float>> outputs;
  for (auto &output_handle : output_handles_) {
    std::vector<int> shape = output_handle->shape();
    size_t numel = 1;
    for (auto dim : shape) numel *= dim;
    std::vector<float> out_data(numel);
    output_handle->CopyToCpu(out_data.data());
    outputs.push_back(std::move(out_data));
  }

  return outputs;
}

absl::Status PaddleInfer::CheckRunMode() {
  // if (
  //     !DISABLE_MKLDNN_MODEL_BL &&
  //     option_.run_mode.rfind("mkldnn", 0) == 0 &&
  //     MKLDNN_BLOCKLIST.count(model_name_) > 0 &&
  //     option_.device_type == "cpu"
  // ) {
  //     std::cout << "The model(" << model_name_ << ") is not supported to run
  //     in MKLDNN mode! Using `paddle` instead!" << std::endl; option_.run_mode
  //     = "paddle";
  // }

  // // check available for model
  // if (_model_name == "LaTeX_OCR_rec" && option_.device_type == "cpu") {
  //     std::string vendor_id_raw = get_cpu_vendor();
  //     if (vendor_id_raw.find("GenuineIntel") != std::string::npos &&
  //     option_.run_mode != "mkldnn") {
  //         std::cout << "Now, the `LaTeX_OCR_rec` model only support `mkldnn`
  //         mode when running on Intel CPU devices. So using `mkldnn` instead."
  //         << std::endl; option_.run_mode = "mkldnn";
  //     }
  // }

  return absl::OkStatus();
}
