#pragma once

namespace myplacement {

// This workspace is allocated physical GPUs 1--4 plus GPU 7.  Keep this
// policy centralized so CLI validation, CPU fallbacks, and CUDA backends never
// disagree about which shared devices are permitted.
constexpr bool isPermittedCudaDevice(int device) {
    return (device >= 1 && device <= 4) || device == 7;
}

}  // namespace myplacement
