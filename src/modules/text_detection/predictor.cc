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

#include "predictor.h"

#include "result.h"
#include "src/common/image_batch_sampler.h"

TextDetPredictor::TextDetPredictor(
    const std::string& model_dir, const std::string& device,
    const bool enable_mkldnn, int batch_size,
    const std::unordered_map<std::string, std::string>& config,

    int limit_side_len, const std::string& limit_type, float thresh,
    float box_thresh, float unclip_ratio, const std::vector<int>& input_shape,
    int max_side_limit)
    : BasePredictor(model_dir, device, enable_mkldnn, batch_size, config),
      limit_side_len_(limit_side_len),
      limit_type_(limit_type),
      thresh_(thresh),
      box_thresh_(box_thresh),
      unclip_ratio_(unclip_ratio),
      input_shape_(input_shape),
      max_side_limit_(max_side_limit) {
  Build();
};

void TextDetPredictor::Build() {
  const auto& pre_tfs = config_.PreProcessOpInfo();
  Register<ReadImage>("Read", pre_tfs.at("DecodeImage.img_mode"));
  Register<DetResizeForTest>(
      "Resize", std::stoi(pre_tfs.at("DetResizeForTest.resize_long")));
  Register<NormalizeImage>("Normalize");
  Register<ToCHWImage>("ToCHW");
  Register<ToBatch>("ToBatch");
  infer_ptr_ = CreateStaticInfer();
  const auto& post_parm = config_.PostProcessOpInfo();
  post_op_["DBPostProcess"] = std::unique_ptr<DBPostProcess>(
      new DBPostProcess(std::stof(post_parm.at("PostProcess.thresh")),
                        std::stof(post_parm.at("PostProcess.box_thresh")),
                        std::stoi(post_parm.at("PostProcess.max_candidates")),
                        std::stof(post_parm.at("PostProcess.unclip_ratio"))));
};

std::vector<std::unique_ptr<BaseCVResult>> TextDetPredictor::Process(
    std::vector<cv::Mat>& batch_data) {
  std::vector<cv::Mat> origin_image = {};
  origin_image.reserve(batch_data.size());
  for (const auto& mat : batch_data) {
    origin_image.push_back(mat.clone());
  }
  auto batch_raw_imgs = pre_op_.at("Read")->Apply(batch_data);
  if (!batch_raw_imgs.ok()) {
    std::cerr << batch_raw_imgs.status().ToString();
  }
  std::vector<int> origin_shape = {(*batch_raw_imgs)[0].rows,
                                   (*batch_raw_imgs)[0].cols};
  auto batch_imgs = pre_op_.at("Resize")->Apply(*batch_raw_imgs);
  if (!batch_imgs.ok()) {
    std::cerr << batch_imgs.status().ToString();
  }
  auto batch_imgs_normalize = pre_op_.at("Normalize")->Apply(*batch_imgs);
  if (!batch_imgs_normalize.ok()) {
    std::cerr << batch_imgs_normalize.status().ToString();
  }

  auto batch_imgs_to_chw = pre_op_.at("ToCHW")->Apply(*batch_imgs_normalize);
  if (!batch_imgs_to_chw.ok()) {
    std::cerr << batch_imgs_to_chw.status().ToString();
  }
  auto batch_imgs_to_batch = pre_op_.at("ToBatch")->Apply(*batch_imgs_to_chw);
  if (!batch_imgs_to_batch.ok()) {
    std::cerr << batch_imgs_to_batch.status().ToString();
  }
  auto infer_result = infer_ptr_->Apply(*batch_imgs_to_batch);
  if (!infer_result.ok()) {
    std::cerr << infer_result.status().ToString();
  }
  auto db_result =
      post_op_.at("DBPostProcess")->Apply((*infer_result)[0], origin_shape);

  if (!db_result.ok()) {
    std::cerr << db_result.status().ToString();
  }
  for (int i = 0; i < db_result.value().size(); i++) {
    TextDetPredictorResult predictor_result;
    predictor_result.input_path["input_path"] = input_path_;
    predictor_result.input_image["input_image"] = origin_image[i];
    predictor_result.dt_polys["dt_polys"] = (*db_result)[i].first;
    predictor_result.dt_scores["dt_score"] = (*db_result)[i].second;
    predictor_result_vec_.push_back(predictor_result);
  }
  std::vector<std::unique_ptr<BaseCVResult>> base_cv_result_ptr_vec = {};
  for (const auto& predictor_result : predictor_result_vec_) {
    std::unique_ptr<BaseCVResult> base_cv_result_ptr =
        std::unique_ptr<BaseCVResult>(new TextDetResult(predictor_result));
    base_cv_result_ptr_vec.emplace_back(std::move(base_cv_result_ptr));
  }
  return std::move(base_cv_result_ptr_vec);
}

std::unique_ptr<BaseBatchSampler> TextDetPredictor::BuildBatchSampler() {
  std::unique_ptr<BaseBatchSampler> sampler =
      std::unique_ptr<BaseBatchSampler>(new ImageBatchSampler(batch_size_));
  return sampler;
}

std::unique_ptr<BaseCVResult> TextDetPredictor::GetResultClass() {
  return nullptr;
}
