// Implementation of PolicyRuntime — see policy_runtime.hpp.
#include "wheeled_rl_controller/policy_runtime.hpp"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace wheeled_rl::onnxruntime
{

bool PolicyRuntime::load(const std::string & model_path)
{
#ifdef HAVE_ONNXRUNTIME
  if (model_path.empty()) return false;
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
  return true;
#else
  (void)model_path;
  return false;
#endif
}

void PolicyRuntime::infer(const std::array<float, kObsDim> & obs,
                          std::array<float, kActionDim> & action)
{
#ifdef HAVE_ONNXRUNTIME
  if (session_) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape[2] = {1, static_cast<int64_t>(kObsDim)};
    auto input = Ort::Value::CreateTensor<float>(mem, const_cast<float *>(obs.data()), kObsDim, shape, 2);
    const char * input_names[] = {input_name_};
    const char * output_names[] = {output_name_};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    const float * act = outputs[0].GetTensorData<float>();
    for (size_t k = 0; k < kActionDim; ++k) action[k] = act[k];
    return;
  }
#endif
  action.fill(0.0f);
}

}  // namespace wheeled_rl::onnxruntime
