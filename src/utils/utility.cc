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

#include "utility.h"

#include <sys/stat.h>

absl::Status Utility::FileExists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0) {
    return absl::OkStatus();
  } else {
    return absl::NotFoundError("File is not exist:" + path);
  }
}
absl::StatusOr<std::map<std::string, std::pair<std::string, std::string> > >
Utility::GetModelPaths(const std::string& model_dir,
                       const std::string& model_file_prefix) {
  std::map<std::string, std::pair<std::string, std::string> > model_paths;
  std::string pd_model_path;

  std::string json_path = model_dir + "/" + model_file_prefix + ".json";
  std::string pdmodel_path = model_dir + "/" + model_file_prefix + ".pdmodel";
  std::string pdiparams_path =
      model_dir + "/" + model_file_prefix + ".pdiparams";
  if (FileExists(json_path).ok()) {
    pd_model_path = json_path;
  } else if (FileExists(pdmodel_path).ok()) {
    pd_model_path = pdmodel_path;
  } else {
    std::cerr << FileExists(json_path).ToString() << " and "
              << FileExists(pdmodel_path).ToString();
  }

  if (pd_model_path.empty()) {
    return absl::NotFoundError(
        "No PaddlePaddle model file (.json or .pdmodel) found!");
  }

  if (FileExists(pdiparams_path).ok()) {
    model_paths["paddle"] = std::make_pair(pd_model_path, pdiparams_path);
  } else {
    return absl::NotFoundError(
        "No PaddlePaddle params file (.pdiparams) found!");
  }

  return model_paths;
}

absl::StatusOr<std::string> Utility::GetConfigPaths(
    const std::string& model_dir, const std::string& model_file_prefix) {
  std::string config_path = "";
  std::string config_path_find = model_dir + "/" + model_file_prefix + ".yml";
  if (FileExists(config_path_find).ok()) {
    config_path = config_path_find;
  } else {
    std::cerr << FileExists(config_path_find).ToString();
  }
  return config_path;
};

std::string Utility::GetCpuVendor() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("vendor_id") != std::string::npos) {
      auto pos = line.find(":");
      if (pos != std::string::npos) {
        return line.substr(pos + 2);
      }
    }
  }
  return "";
};

void Utility::WriteBatchMatToTxt(const cv::Mat& batch,
                                 const std::string& filename) {
  // 检查维度和类型
  if (batch.dims != 4 || batch.type() != CV_32F) {
    std::cerr << "Input must be 4D CV_32F Mat." << std::endl;
    return;
  }

  const int batch_size = batch.size[0];
  const int channels = batch.size[1];
  const int rows = batch.size[2];
  const int cols = batch.size[3];

  std::ofstream fout(filename);
  if (!fout.is_open()) {
    std::cerr << "Cannot open file for writing: " << filename << std::endl;
    return;
  }

  // 顺序为 NCHW
  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < rows; ++h) {
        // 每一行的首地址
        const float* row_ptr = batch.ptr<float>(n, c, h);
        for (int w = 0; w < cols; ++w) {
          fout << row_ptr[w] << "\n";
        }
      }
    }
  }

  fout.close();
  std::cout << "Write done: " << filename << std::endl;
}

void Utility::WriteBatchMatToTxt_X(const cv::Mat& mat,
                                   const std::string& filename) {
  std::ofstream fout(filename);
  if (!fout.is_open()) {
    std::cerr << "Cannot open file for writing: " << filename << std::endl;
    return;
  }

  fout << "Dimensions: " << mat.dims << "\n";
  for (int i = 0; i < mat.dims; ++i) {
    fout << "Size[" << i << "]: " << mat.size[i] << "\n";
  }
  fout << "Type: " << mat.type() << "\n\n";

  size_t total_elements = mat.total();

  switch (mat.type()) {
    case CV_8U:
      for (size_t i = 0; i < total_elements; ++i)
        fout << static_cast<int>(mat.ptr<uchar>()[i]) << "\n";
      break;
    case CV_8S:
      for (size_t i = 0; i < total_elements; ++i)
        fout << static_cast<int>(mat.ptr<char>()[i]) << "\n";
      break;
    case CV_16U:
      for (size_t i = 0; i < total_elements; ++i)
        fout << mat.ptr<ushort>()[i] << "\n";
      break;
    case CV_16S:
      for (size_t i = 0; i < total_elements; ++i)
        fout << mat.ptr<short>()[i] << "\n";
      break;
    case CV_32S:
      for (size_t i = 0; i < total_elements; ++i)
        fout << mat.ptr<int>()[i] << "\n";
      break;
    case CV_32F:
      for (size_t i = 0; i < total_elements; ++i)
        fout << mat.ptr<float>()[i] << "\n";
      break;
    case CV_64F:
      for (size_t i = 0; i < total_elements; ++i)
        fout << mat.ptr<double>()[i] << "\n";
      break;
    default:
      std::cerr << "Unsupported mat type: " << mat.type() << std::endl;
      fout.close();
      return;
  }

  fout.close();
  std::cout << "Write done: " << filename << std::endl;
}

void Utility::PrintShape(const cv::Mat& img) {
  for (int i = 0; i < img.dims; i++) {
    std::cout << img.size[i] << " ";
  }
  std::cout << std::endl;
}

absl::Status Utility::CreateDirectory(const std::string& path) {
#ifdef _WIN32
  int ret = _mkdir(path.c_str());
#else
  int ret = mkdir(path.c_str(), 0755);
#endif
  if (ret == 0) {
    return absl::OkStatus();
  }
  if (errno == EEXIST) {
    return absl::OkStatus();
  }
  return absl::ErrnoToStatus(errno, "Failed to create directory: " + path);
}

absl::Status Utility::CreatePath(const std::string& path) {
  std::vector<std::string> paths;
  std::string tmp;
  for (size_t i = 0; i < path.size(); ++i) {
    tmp += path[i];
    if (path[i] == '/' || path[i] == '\\') {
      paths.push_back(tmp);
    }
  }
  if (!tmp.empty() && tmp.back() != '/' && tmp.back() != '\\')
    paths.push_back(tmp);

  std::string current;
  for (size_t i = 0; i < paths.size(); ++i) {
    current += paths[i];
    absl::Status status = CreateDirectory(current);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::Status Utility::CreateFile(const std::string& filepath) {
  std::ifstream infile(filepath.c_str());
  if (infile.good()) {
    return absl::OkStatus();
  }

  std::ofstream outfile(filepath.c_str(), std::ios::out | std::ios::trunc);
  if (!outfile.is_open()) {
    return absl::InternalError("Failed to create file: " + filepath);
  }

  outfile.close();
  return absl::OkStatus();
}

const std::unordered_set<std::string> Utility::MKLDNN_BLOCKLIST = {
    "LaTeX_OCR_rec",
    "PP-FormulaNet-L",
    "PP-FormulaNet-S",
    "UniMERNet",
    "UVDoc",
    "Cascade-MaskRCNN-ResNet50-FPN",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
    "Mask-RT-DETR-M",
    "Mask-RT-DETR-S",
    "MaskRCNN-ResNeXt101-vd-FPN",
    "MaskRCNN-ResNet101-FPN",
    "MaskRCNN-ResNet101-vd-FPN",
    "MaskRCNN-ResNet50-FPN",
    "MaskRCNN-ResNet50-vd-FPN",
    "MaskRCNN-ResNet50",
    "SOLOv2",
    "PP-TinyPose_128x96",
    "PP-TinyPose_256x192",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "Co-DINO-Swin-L",
    "Co-Deformable-DETR-Swin-T",
    "FasterRCNN-ResNeXt101-vd-FPN",
    "FasterRCNN-ResNet101-FPN",
    "FasterRCNN-ResNet101",
    "FasterRCNN-ResNet34-FPN",
    "FasterRCNN-ResNet50-FPN",
    "FasterRCNN-ResNet50-vd-FPN",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "FasterRCNN-ResNet50",
    "FasterRCNN-Swin-Tiny-FPN",
    "MaskFormer_small",
    "MaskFormer_tiny",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "SLANet",
    "SLANet_plus",
    "YOWO",
    "SAM-H_box",
    "SAM-H_point",
    "PP-FormulaNet_plus-L",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-S"};
