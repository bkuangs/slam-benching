# Minimal stand-in for find_package(Torch) used by the slam-benching SP_SLAM3 image.
# LibTorch's own TorchConfig.cmake requires a CUDA toolkit (nvcc, cuDNN headers) for its CUDA
# build even though SP_SLAM3 compiles no CUDA code; SP_SLAM3 only uses the LibTorch C++ API, so
# linking the prebuilt libraries directly is enough. libtorch_cuda / libc10_cuda are linked with
# --no-as-needed so the CUDA backend registers itself even without a direct symbol reference.
if(NOT DEFINED LIBTORCH_ROOT)
  set(LIBTORCH_ROOT "/opt/libtorch")
endif()
set(TORCH_INCLUDE_DIRS ${LIBTORCH_ROOT}/include ${LIBTORCH_ROOT}/include/torch/csrc/api/include)
set(TORCH_LIBRARIES
    -Wl,--no-as-needed
    ${LIBTORCH_ROOT}/lib/libtorch_cuda.so ${LIBTORCH_ROOT}/lib/libc10_cuda.so
    -Wl,--as-needed
    ${LIBTORCH_ROOT}/lib/libtorch.so ${LIBTORCH_ROOT}/lib/libtorch_cpu.so ${LIBTORCH_ROOT}/lib/libc10.so)
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
add_definitions(${TORCH_CXX_FLAGS})
set(Torch_FOUND TRUE)
