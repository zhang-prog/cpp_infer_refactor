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

#include "result.h"

#include <algorithm>
#include <fstream>
#include <random>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;

void OCRResult::SaveToImg(const std::string& save_path) {
  cv::Mat image = pipeline_result_.doc_preprocessor_res.output_image;
  auto texts = pipeline_result_.rec_texts;
  std::vector<std::vector<cv::Point>> boxes;
  std::vector<std::vector<cv::Point2f>> boxes_float =
      pipeline_result_.rec_polys;
  for (const auto& floatPolygon : pipeline_result_.rec_polys) {
    std::vector<cv::Point> intPolygon;
    for (const auto& point : floatPolygon) {
      intPolygon.push_back(cv::Point(cvRound(point.x), cvRound(point.y)));
    }
    boxes.push_back(intPolygon);
  }

  if (image.empty()) {
    INFOE("Input image is empty.");
  }

  int h = image.rows;
  int w = image.cols;

  // cv::Mat image_rgb = image;
  // cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

  cv::Mat img_left = image.clone();

  cv::Mat img_right(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100, 100);

  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto& box = boxes[i];
    const auto& box_float = boxes_float[i];
    const auto& text = texts[i];

    cv::Scalar color(dis(gen), dis(gen), dis(gen));

    if (box.size() > 4) {
      cv::fillPoly(img_left, std::vector<std::vector<cv::Point>>{box}, color);
      // Further processing for rotated rectangles can be added here
    } else {
      cv::fillPoly(img_left, std::vector<std::vector<cv::Point>>{box}, color);
    }

    cv::Mat img_right_text = DrawBoxTextFine(
        cv::Size(w, h), box_float, text);  // Placeholder for drawing text
    cv::imwrite("img_right_text.jpg", img_right_text);
    cv::polylines(img_right_text, box, true, color, 1);

    cv::Mat gray, mask;
    cv::cvtColor(img_right_text, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 128, 255, cv::THRESH_BINARY_INV);
    img_right_text.copyTo(img_right, mask);
  }

  cv::Mat blended;
  cv::addWeighted(image, 0.5, img_left, 0.5, 0, blended);

  cv::Mat img_show(h, w * 2, CV_8UC3, cv::Scalar(255, 255, 255));
  blended.copyTo(img_show(cv::Rect(0, 0, w, h)));
  img_right.copyTo(img_show(cv::Rect(w, 0, w, h)));

  auto model_setting = pipeline_result_.model_setting;
  std::unordered_map<std::string, cv::Mat> res_img_dict;
  res_img_dict["ocr_res_img"] = img_show;
  if (model_setting["useDocPreprocessor"]) {
    res_img_dict["useDocPreprocessor"] =
        pipeline_result_.doc_preprocessor_res.image_all;
  }

  cv::imwrite("gsj.jpg", img_show);
}

cv::Mat OCRResult::DrawBoxTextFine(const cv::Size& imgSize,
                                   const std::vector<cv::Point2f>& box,
                                   const std::string& txt) {
  auto calculateDistance = [](const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
  };

  int boxHeight = static_cast<int>(calculateDistance(box[0], box[3]));
  int boxWidth = static_cast<int>(calculateDistance(box[0], box[1]));

  cv::Mat imgText = cv::Mat::zeros(boxHeight, boxWidth, CV_8UC3);
  imgText.setTo(cv::Scalar(255, 255, 255));

  if (!txt.empty()) {
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 1;
    int baseline = 0;

    // 计算文本尺寸以居中
    cv::Size textSize =
        cv::getTextSize(txt, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrg((boxWidth - textSize.width) / 2,
                      (boxHeight + textSize.height) / 2);

    // 绘制文本
    cv::putText(imgText, txt, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 0),
                thickness, cv::LINE_AA);
  }

  // 定义透视变换的源点和目标点
  std::vector<cv::Point2f> pts1 = {{0.0, 0.0},
                                   {(float)boxWidth, 0.0},
                                   {(float)boxWidth, (float)boxHeight},
                                   {0.0, (float)boxHeight}};
  cv::Mat M = cv::getPerspectiveTransform(pts1, box);

  cv::Mat imgRightText;
  cv::warpPerspective(imgText, imgRightText, M, imgSize, cv::INTER_NEAREST,
                      cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

  return imgRightText;
}

void OCRResult::Print() const { int a = 1; }
void OCRResult::SaveToJson(const std::string& save_path) const { int a = 1; }
