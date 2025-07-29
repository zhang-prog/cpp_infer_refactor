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

#ifndef YAML_CONFIG_H_
#define YAML_CONFIG_H_

#include <yaml-cpp/yaml.h>

#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

class YamlConfig {
 public:
  YamlConfig(const std::unordered_map<std::string, std::string>& data)
      : data_(data) {}
  YamlConfig(const std::string& file_name);
  ~YamlConfig() = default;

  absl::Status LoadFile(const std::string& file_name);
  void Init();
  std::unordered_map<std::string, std::string> PreProcessOpInfo() {
    return pre_process_op_info_;
  };
  std::unordered_map<std::string, std::string> PostProcessOpInfo() {
    return post_process_op_info_;
  };
  absl::StatusOr<std::string> GetString(const std::string& key) const;
  absl::StatusOr<int> GetInt(const std::string& key) const;
  absl::StatusOr<double> GetDouble(const std::string& key) const;
  absl::StatusOr<bool> GetBool(const std::string& key) const;

  absl::Status HasKey(const std::string& key) const;

  absl::Status PrintAll() const;
  absl::Status PrintWithPrefix(const std::string& prefix) const;
  absl::Status FindPreProcessOp(
      const std::string& prefix = "PreProcess.transform_ops[0]") const;
  std::unordered_map<std::string, std::string> Data() { return data_; }

 private:
  void ParseNode(const YAML::Node& node, const std::string& prefix = "");
  std::unordered_map<std::string, std::string> data_;
  std::unordered_map<std::string, std::string> pre_process_op_info_;
  std::unordered_map<std::string, std::string> post_process_op_info_;
};

#endif  // YAML_CONFIG_H_
