// ONNX policy runtime — ORT session wrapper (the onnxruntime module role).
// Guarded by HAVE_ONNXRUNTIME; without it the controller runs the control
// path only (holds zero actions).
#ifndef WHEELED_RL_CONTROLLER__POLICY_RUNTIME_HPP_
#define WHEELED_RL_CONTROLLER__POLICY_RUNTIME_HPP_

#include <array>
#include <memory>
#include <string>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace wheeled_rl::onnxruntime
{

constexpr size_t kObsDim = 35;    // CONTRACT.md section 2
constexpr size_t kActionDim = 6;  // CONTRACT.md section 3

class PolicyRuntime
{
public:
  /// Load the ONNX policy; single-threaded to keep the 500 Hz loop schedulable.
  bool load(const std::string & model_path);

  bool ready() const { return static_cast<bool>(session_); }

  /// obs[35] -> action[6] (mean policy, no sampling noise).
  void infer(const std::array<float, kObsDim> & obs, std::array<float, kActionDim> & action);

private:
#ifdef HAVE_ONNXRUNTIME
  Ort::Env env_{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "wheeled_rl"};
  std::unique_ptr<Ort::Session> session_;
  const char * input_name_ = "obs";
  const char * output_name_ = "actions";
#endif
};

}  // namespace wheeled_rl::onnxruntime

#endif  // WHEELED_RL_CONTROLLER__POLICY_RUNTIME_HPP_
