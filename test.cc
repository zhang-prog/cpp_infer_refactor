#include <paddle_inference_api.h>

#include <opencv2/opencv.hpp>

#include "src/base/base_predictor.h"
#include "src/modules/text_detection/predictor.h"

int main() {
  BasePredictor* infer =
      new TextDetPredictor("/workspace/cpp_infer_refactor/PP-OCRv5_mobile_det");
  auto outputs =
      infer->Predict("/workspace/cpp_infer_refactor/pp_structure_v3_demo.png");
  for (auto& output : outputs) {
    output->Print();
    output->SaveToImg("./output/");
    output->SaveToJson("./output/res.json");
  }
  return 0;
}
