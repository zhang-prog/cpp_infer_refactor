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

#ifndef FUNC_REGISTER_H_
#define FUNC_REGISTER_H_

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

class BaseProcessor {
 public:
  virtual ~BaseProcessor() = default;
  virtual void Apply() const = 0;
};

class FuncRegister {
 public:
  template <typename T, typename... Args>
  void Register(const std::string &key, Args &&...args);

  void Apply(const std::string &key);

 private:
  std::unordered_map<std::string, std::unique_ptr<BaseProcessor>> register_map_;
};

template <typename T, typename... Args>
void FuncRegister::Register(const std::string &key, Args &&...args) {
  auto instance = std::make_unique<T>(std::forward<Args>(args)...);
  register_map_[key] = std::move(instance);
}

#endif  // FUNC_REGISTER_H_
