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

#include <fstream>
#include <string>

#include "src/utils/utility.h"
#include "third_party/nlohmann/json.hpp"

using json = nlohmann::json;
void TextDetResult::SaveToImg(const std::string& save_path) const {
  std::string file_name = "output_image.png";
  std::string full_path = save_path;
  if (save_path.back() != '/' && save_path.back() != '\\') {
    full_path += '/';
  }
  auto result = Utility::CreatePath(full_path);
  full_path += file_name;
  if (!result.ok()) {
    std::cerr << result.ToString();
  }

  cv::Mat img = predictor_result_.input_image.at("input_image").clone();

  const auto& dt_polys = predictor_result_.dt_polys.at("dt_polys");
  for (const auto& poly : dt_polys) {
    std::vector<cv::Point> pts;
    for (const auto& pt : poly) {
      pts.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
    }

    const cv::Point* pts_ptr = pts.data();
    int npts = pts.size();
    cv::polylines(img, &pts_ptr, &npts, 1, true, cv::Scalar(0, 0, 255),
                  2);  // 绿色框, 宽度2
  }

  // 保存
  bool success = cv::imwrite(full_path, img);
  if (!success) {
    std::cerr << "Error: Failed to write the image to " << full_path
              << std::endl;
  }
}

void TextDetResult::Print() const {
  std::cout << "{\n  \"res\": {" << std::endl;

  // Print input_path
  std::cout << "    \"input_path\": {" << std::endl;
  for (const auto& pair : predictor_result_.input_path) {
    std::cout << "      \"" << pair.first << "\": \"" << pair.second << "\","
              << std::endl;
  }
  std::cout << "    }," << std::endl;

  // Print dt_polys
  std::cout << "    \"dt_polys\": [" << std::endl;
  for (const auto& pair : predictor_result_.dt_polys) {
    std::cout << "      {\"" << pair.first << "\": [" << std::endl;
    for (const auto& polygon : pair.second) {
      std::cout << "        [";
      for (size_t i = 0; i < polygon.size(); ++i) {
        std::cout << "[" << static_cast<int>(polygon[i].x) << ", "
                  << static_cast<int>(polygon[i].y) << "]";
        if (i < polygon.size() - 1) std::cout << ", ";
      }
      std::cout << "]," << std::endl;
    }
    std::cout << "      ]}," << std::endl;
  }
  std::cout << "    ]," << std::endl;

  // Print dt_scores
  std::cout << "    \"dt_scores\": [" << std::endl;
  for (const auto& pair : predictor_result_.dt_scores) {
    std::cout << "      {\"" << pair.first << "\": [";
    for (size_t i = 0; i < pair.second.size(); ++i) {
      std::cout << pair.second[i];
      if (i < pair.second.size() - 1) std::cout << ", ";
    }
    std::cout << "]}" << std::endl;
  }
  std::cout << "    ]" << std::endl;

  std::cout << "  }\n}" << std::endl;
}

void TextDetResult::SaveToJson(const std::string& save_path) const {
  nlohmann::ordered_json j;

  for (const auto& pair : predictor_result_.input_path) {
    j[pair.first] = pair.second;
  }
  j["page_index"] = nullptr;  //********
  for (const auto& pair : predictor_result_.dt_polys) {
    json polys_json = json::array();
    for (const auto& polygon : pair.second) {
      json poly_json = json::array();
      for (const auto& point : polygon) {
        poly_json.push_back(
            {static_cast<int>(point.x), static_cast<int>(point.y)});
      }
      polys_json.push_back(poly_json);
    }
    j[pair.first] = polys_json;
  }

  // Convert dt_scores to JSON
  for (const auto& pair : predictor_result_.dt_scores) {
    j[pair.first] = pair.second;
  }
  auto result = Utility::CreateFile(save_path);
  if (!result.ok()) {
    std::cerr << result.ToString();
  }
  // Write JSON to file
  std::ofstream file(save_path);
  if (file.is_open()) {
    file << j.dump(4);
    file.close();
  } else {
    std::cerr << "Could not open file for writing: " << save_path << std::endl;
  }
}
