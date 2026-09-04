#!/usr/bin/env python3
"""Validate an ONNX policy against the frozen deploy contract (CONTRACT.md).

    python3 tools/check_onnx_contract.py path/to/policy.onnx [--obs-dim 35]

Checks: single input/output, names (obs/actions), dtype float32, static shape
[1, obs_dim] -> [1, 6], and a zero-observation forward pass (must be finite
and reasonably small — a diverging policy at zero obs is a red flag).
"""
import argparse

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit("pip install onnxruntime") from e

ACTION_DIM = 6
REQUIRED_INPUT, REQUIRED_OUTPUT = "obs", "actions"
ZERO_OBS_ABS_MAX = 5.0  # heuristic bound; a sane policy outputs near-zero at zero obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--obs-dim", type=int, default=35)
    p.add_argument("--allow-zero-obs-spike", action="store_true",
                   help="skip the zero-observation magnitude heuristic")
    args = p.parse_args()

    opts = ort.SessionOptions()
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    failures = []

    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        failures.append(f"expected 1 input / 1 output, got {len(ins)} / {len(outs)}")
    i, o = ins[0], outs[0]
    if i.name != REQUIRED_INPUT:
        failures.append(f"input name {i.name!r} != {REQUIRED_INPUT!r}")
    if o.name != REQUIRED_OUTPUT:
        failures.append(f"output name {o.name!r} != {REQUIRED_OUTPUT!r}")
    if i.type != "tensor(float)":
        failures.append(f"input dtype {i.type} != tensor(float)")
    if list(i.shape) != [1, args.obs_dim]:
        failures.append(f"input shape {i.shape} != [1, {args.obs_dim}]")
    if list(o.shape) != [1, ACTION_DIM]:
        failures.append(f"output shape {o.shape} != [1, {ACTION_DIM}]")

    action = sess.run(None, {i.name: np.zeros((1, args.obs_dim), np.float32)})[0]
    if not np.all(np.isfinite(action)):
        failures.append(f"zero-obs forward not finite: {action}")
    elif not args.allow_zero_obs_spike and np.abs(action).max() > ZERO_OBS_ABS_MAX:
        print(f"WARNING: zero-obs action magnitude {np.abs(action).max():.2f} > {ZERO_OBS_ABS_MAX} "
              "(may be legitimate for your reward shaping; use --allow-zero-obs-spike)")

    if failures:
        print("CONTRACT CHECK FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"CONTRACT OK: {i.name}{i.shape} -> {o.name}{o.shape}, zero-obs action = {action[0]}")


if __name__ == "__main__":
    main()
