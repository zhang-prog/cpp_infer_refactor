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

#include "image_batch_sampler.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <iostream>

const std::set<std::string> ImageBatchSampler::kImgSuffixes = {"jpg", "png",
                                                               "jpeg", "bmp"};

ImageBatchSampler::ImageBatchSampler(int batch_size)
    : BaseBatchSampler(batch_size) {}

absl::StatusOr<std::vector<std::vector<cv::Mat> > >
ImageBatchSampler::SampleFromString(const std::string& input) {
  std::vector<std::string> inputs = {input};
  return SampleFromVector(inputs);
}

absl::StatusOr<std::vector<std::vector<cv::Mat> > >
ImageBatchSampler::SampleFromVector(const std::vector<std::string>& inputs) {
  std::vector<std::vector<cv::Mat> > result;
  std::vector<cv::Mat> current_batch;

  for (size_t i = 0; i < inputs.size(); ++i) {
    const std::string& input = inputs[i];

    if (IsDirectory(input)) {
      absl::StatusOr<std::vector<std::string> > files_result =
          GetFilesList(input);
      if (!files_result.ok()) {
        return files_result.status();
      }

      absl::StatusOr<std::vector<std::vector<cv::Mat> > > sub_result =
          SampleFromVector(files_result.value());
      if (!sub_result.ok()) {
        return sub_result.status();
      }

      const std::vector<std::vector<cv::Mat> >& sub_batches =
          sub_result.value();
      for (size_t j = 0; j < sub_batches.size(); ++j) {
        result.push_back(sub_batches[j]);
      }
    } else if (IsImageFile(input)) {
      if (!FileExists(input)) {
        return absl::NotFoundError("File not found: " + input);
      }

      absl::StatusOr<cv::Mat> image_result = LoadImage(input);
      if (!image_result.ok()) {
        return image_result.status();
      }

      current_batch.push_back(image_result.value());

      if (static_cast<int>(current_batch.size()) == batch_size_) {
        result.push_back(current_batch);
        current_batch.clear();
      }
    } else {
      std::cerr << "Unsupported file type: " << input << std::endl;
    }
  }

  if (!current_batch.empty()) {
    result.push_back(current_batch);  // last batch
  }

  return result;
}

absl::StatusOr<std::vector<std::string> > ImageBatchSampler::GetFilesList(
    const std::string& path) {
  if (!FileExists(path)) {
    return absl::NotFoundError("Path not found: " + path);
  }

  std::vector<std::string> file_list;

  if (!IsDirectory(path)) {
    if (IsImageFile(path)) {
      file_list.push_back(path);
    }
  } else {
    GetFilesRecursive(path, file_list);
  }

  if (file_list.empty()) {
    return absl::NotFoundError("No image files found in path: " + path);
  }

  std::sort(file_list.begin(), file_list.end());
  return file_list;
}

absl::StatusOr<cv::Mat> ImageBatchSampler::LoadImage(
    const std::string& file_path) {
  cv::Mat image = cv::imread(file_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    return absl::InvalidArgumentError("Failed to load image: " + file_path);
  }
  return image;
}

void ImageBatchSampler::GetFilesRecursive(
    const std::string& dir_path, std::vector<std::string>& file_list) const {
  DIR* dir = opendir(dir_path.c_str());
  if (dir == NULL) {
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }

    std::string full_path = dir_path + "/" + name;

    if (IsDirectory(full_path)) {
      GetFilesRecursive(full_path, file_list);
    } else if (IsImageFile(full_path)) {
      file_list.push_back(full_path);
    }
  }

  closedir(dir);
}

bool ImageBatchSampler::IsImageFile(const std::string& file_path) const {
  std::string extension = GetFileExtension(file_path);
  std::string lower_ext = ToLower(extension);
  return kImgSuffixes.find(lower_ext) != kImgSuffixes.end();
}

bool ImageBatchSampler::IsDirectory(const std::string& path) const {
  struct stat path_stat;
  if (stat(path.c_str(), &path_stat) != 0) {
    return false;
  }
  return S_ISDIR(path_stat.st_mode);
}

bool ImageBatchSampler::FileExists(const std::string& path) const {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

std::string ImageBatchSampler::GetFileExtension(
    const std::string& file_path) const {
  size_t pos = file_path.find_last_of('.');
  if (pos == std::string::npos || pos == file_path.length() - 1) {
    return "";
  }
  return file_path.substr(pos + 1);
}

std::string ImageBatchSampler::ToLower(const std::string& str) const {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}
