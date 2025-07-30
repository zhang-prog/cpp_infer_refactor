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

#include "yaml_config.h"

#include <iostream>
#include <sstream>

#include "absl/strings/str_cat.h"

YamlConfig::YamlConfig(const std::string& model_dir) {
  auto status_get = GetConfigYamlPaths(model_dir);
  if (!status_get.ok()) {
    std::cerr << "Could find files with the .yaml or .yml in " + model_dir +
                     " !"
              << status_get.ToString() << std::endl;
  }
  auto status = LoadYamlFile();
  if (!status.ok()) {
    std::cerr << "Failed to load config: " << status.ToString() << std::endl;
  }
  Init();
}

absl::Status YamlConfig::GetConfigYamlPaths(const std::string& model_dir) {
  std::string config_path_yml =
      model_dir + "/" + Utility::MODEL_FILE_PREFIX + ".yml";
  std::string config_path_yaml =
      model_dir + "/" + Utility::MODEL_FILE_PREFIX + ".yaml";
  if (Utility::FileExists(config_path_yml).ok()) {
    config_yaml_path_ = config_path_yml;
    return absl::OkStatus();
  } else if (Utility::FileExists(config_path_yaml).ok()) {
    config_yaml_path_ = config_path_yaml;
    return absl::OkStatus();
  } else {
    return absl::NotFoundError("file is not exist!");
  }
};

absl::Status YamlConfig::LoadYamlFile() {
  try {
    YAML::Node config = YAML::LoadFile(config_yaml_path_);
    ParseNode(config);
    return absl::OkStatus();
  } catch (const YAML::BadFile& e) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open YAML file: ", config_yaml_path_));
  } catch (const YAML::ParserException& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse YAML file: ", e.what()));
  } catch (const YAML::Exception& e) {
    return absl::InternalError(absl::StrCat("YAML error: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrCat("Unexpected error: ", e.what()));
  }
}

void YamlConfig::Init() {
  for (const auto& info : data_) {
    if (info.first.find("DecodeImage.channel_first") != std::string::npos) {
      pre_process_op_info_["DecodeImage.channel_first"] = info.second;
    } else if (info.first.find("DecodeImage.img_mode") != std::string::npos) {
      pre_process_op_info_["DecodeImage.img_mode"] = info.second;
    } else if (info.first.find("DetLabelEncode") != std::string::npos) {
      pre_process_op_info_["DetLabelEncode"] = info.second;
    } else if (info.first.find("DetResizeForTest.resize_long") !=
               std::string::npos) {
      pre_process_op_info_["DetResizeForTest.resize_long"] = info.second;
    } else if (info.first.find("NormalizeImage.mean") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.mean"] = info.second;
    } else if (info.first.find("NormalizeImage.order") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.order"] = info.second;
    } else if (info.first.find("NormalizeImage.scale") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.scale"] = info.second;
    } else if (info.first.find("NormalizeImage.std") != std::string::npos) {
      pre_process_op_info_["NormalizeImage.std"] = info.second;
    } else if (info.first.find("ToCHWImage") != std::string::npos) {
      pre_process_op_info_["ToCHWImage"] = info.second;
    } else if (info.first.find("KeepKeys.keep_keys") != std::string::npos) {
      pre_process_op_info_["KeepKeys.keep_keys"] = info.second;
    } else if (info.first.find("PostProcess.name") != std::string::npos) {
      post_process_op_info_["PostProcess.name"] = info.second;
    } else if (info.first.find("PostProcess.thresh") != std::string::npos) {
      post_process_op_info_["PostProcess.thresh"] = info.second;
    } else if (info.first.find("PostProcess.box_thresh") != std::string::npos) {
      post_process_op_info_["PostProcess.box_thresh"] = info.second;
    } else if (info.first.find("PostProcess.max_candidates") !=
               std::string::npos) {
      post_process_op_info_["PostProcess.max_candidates"] = info.second;
    } else if (info.first.find("PostProcess.unclip_ratio") !=
               std::string::npos) {
      post_process_op_info_["PostProcess.unclip_ratio"] = info.second;
    }
  }
}

void YamlConfig::ParseNode(const YAML::Node& node, const std::string& prefix) {
  if (node.IsMap()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      std::string key = prefix.empty()
                            ? it->first.as<std::string>()
                            : prefix + "." + it->first.as<std::string>();
      ParseNode(it->second, key);
    }
  } else if (node.IsSequence()) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < node.size(); ++i) {
      std::string index_key = prefix + "[" + std::to_string(i) + "]";
      if (node[i].IsScalar()) {
        data_[index_key] = node[i].as<std::string>();
        if (i > 0) ss << ", ";
        ss << node[i].as<std::string>();
      } else {
        ParseNode(node[i], index_key);
      }
    }
    ss << "]";
    data_[prefix] = ss.str();
  } else if (node.IsScalar()) {
    data_[prefix] = node.as<std::string>();
  } else if (node.IsNull()) {
    data_[prefix] = "null";
  }
}

absl::StatusOr<std::string> YamlConfig::GetString(
    const std::string& key) const {
  auto it = data_.find(key);
  if (it != data_.end()) {
    return it->second;
  }
  return absl::NotFoundError(absl::StrCat("Key not found: ", key));
}

absl::StatusOr<int> YamlConfig::GetInt(const std::string& key) const {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return absl::NotFoundError(absl::StrCat("Key not found: ", key));
  }
  try {
    return std::stoi(it->second);
  } catch (const std::invalid_argument&) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid integer value for key '", key, "': ", it->second));
  } catch (const std::out_of_range&) {
    return absl::OutOfRangeError(absl::StrCat(
        "Integer value out of range for key '", key, "': ", it->second));
  }
}

absl::StatusOr<double> YamlConfig::GetDouble(const std::string& key) const {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return absl::NotFoundError(absl::StrCat("Key not found: ", key));
  }
  try {
    return std::stod(it->second);
  } catch (const std::invalid_argument&) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid double value for key '", key, "': ", it->second));
  } catch (const std::out_of_range&) {
    return absl::OutOfRangeError(absl::StrCat(
        "Double value out of range for key '", key, "': ", it->second));
  }
}

absl::StatusOr<bool> YamlConfig::GetBool(const std::string& key) const {
  auto it = data_.find(key);
  if (it == data_.end()) {
    return absl::NotFoundError(absl::StrCat("Key not found: ", key));
  }
  std::string value = it->second;
  if (value == "true" || value == "1" || value == "yes" || value == "True" ||
      value == "Yes" || value == "YES") {
    return true;
  }
  if (value == "false" || value == "0" || value == "no" || value == "False" ||
      value == "No" || value == "NO") {
    return false;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid bool value for key '", key, "': ", value));
}

absl::Status YamlConfig::HasKey(const std::string& key) const {
  if (data_.find(key) != data_.end()) {
    return absl::OkStatus();
  }
  return absl::NotFoundError(absl::StrCat("Key not found: ", key));
}

absl::Status YamlConfig::PrintAll() const {
  for (const auto& it : data_) {
    std::cout << it.first << ": " << it.second << std::endl;
  }
  return absl::OkStatus();
}

absl::Status YamlConfig::PrintWithPrefix(const std::string& prefix) const {
  for (const auto& it : data_) {
    if (it.first.find(prefix) == 0) {
      std::cout << it.first << ": " << it.second << std::endl;
    }
  }
  return absl::OkStatus();
}

absl::Status YamlConfig::FindPreProcessOp(const std::string& prefix) const {
  std::unordered_map<std::string, std::string> pre_process_op_info{};
  for (const auto& it : data_) {
    if (it.first.find(prefix) == 0) {
      std::cout << it.first << ": " << it.second << std::endl;
    }
  }
  return absl::OkStatus();
}
