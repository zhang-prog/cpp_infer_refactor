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

#ifndef PREDICTOR_H_
#define PREDICTOR_H_
#include "processors.h"
#include "src/base/base_batch_sampler.h"
#include "src/base/base_cv_result.h"
#include "src/base/base_predictor.h"

struct TextDetPredictorResult {
  std::unordered_map<std::string, std::string> input_path = {};
  std::unordered_map<std::string, cv::Mat> input_image = {};
  std::unordered_map<std::string, std::vector<std::vector<cv::Point2f>>>
      dt_polys = {};
  std::unordered_map<std::string, std::vector<float>> dt_scores = {};
};

class TextDetPredictor : public BasePredictor {
 public:
  TextDetPredictor(  //***********暂时手动给值**********
      const std::string &model_dir,
      const std::string &device = std::string("cpu"),
      const bool enable_mkldnn = false, int batch_size = 1,
      const std::unordered_map<std::string, std::string> &config = {},

      int limit_side_len = 64, const std::string &limit_type = "min",
      float thresh = 0.3, float box_thresh = 0.6, float unclip_ratio = 1.5,
      const std::vector<int> &input_shape = std::vector<int>(),
      int max_side_limit = 4000);
  void Build();
  absl::Status RegisterFunc();
  absl::Status BuildReadImage();
  absl::Status BuildResize();
  absl::Status BuildNormlization();
  absl::Status BuildCHW();
  absl::Status BuildPostProcess();
  std::unique_ptr<BaseBatchSampler> BuildBatchSampler() override;
  std::vector<std::unique_ptr<BaseCVResult>> Process(
      std::vector<cv::Mat> &batch_data) override;
  std::unique_ptr<BaseCVResult> GetResultClass() override;
  template <typename T, typename... Args>
  void Register(const std::string &key, Args &&...args) {
    auto instance = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    pre_op_[key] = std::move(instance);
  };

 private:
  int limit_side_len_;
  std::string limit_type_;
  float thresh_;
  float box_thresh_;
  float unclip_ratio_;
  std::vector<int> input_shape_;
  int max_side_limit_;
  // std::unique_ptr<FuncRegister> func_register_;
  std::unordered_map<std::string, std::unique_ptr<BaseProcessor>> pre_op_;
  std::unordered_map<std::string, std::unique_ptr<DBPostProcess>> post_op_;
  std::unique_ptr<PaddleInfer> infer_ptr_;
  std::vector<TextDetPredictorResult> predictor_result_vec_;
};

#endif
