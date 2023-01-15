from setuptools import setup, Extension
from torch.utils import cpp_extension
from os import environ

# Constants and env variables
CUSPARSELT_PATH = environ['CUSPARSELT_PATH']
CUDA_TOOLKIT = '/usr/local/cuda-11.2' # environ['CUDA_TOOLKIT']
OS_ARCH_NVRTC = 'x86_64-linux'
NVRTC_SHARED = f'${CUDA_TOOLKIT}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so'

nvcc_includes = [f'-I$(CUDA_TOOLKIT)/include', f'-I${CUSPARSELT_PATH}/include']
include_dirs = [
      cpp_extension.include_paths(cuda=True),
      f'{CUDA_TOOLKIT}/include',
]
nvcc_libs = [f'-L${CUSPARSELT_PATH}/lib', '-lcusparseLt', '-lcudart', '-lcusparse', '-ldl', NVRTC_SHARED]
ext_modules = []
ext_modules.append(
      cpp_extension.CUDAExtension(
            name='ops_cpp',
            sources=[
                  # 'spmma_example.h',
                  # 'spmma_example.cpp',
                  'ops_cuda.cpp',
                  'ops_cuda_kernel.cu',
            ],
            include_dirs=include_dirs,
            # extra_compile_args={'cxx': [],
            #                     'nvcc': ['--std=c++14', # '-O2',
            #                         #      '-gencode', 'arch=compute_70,code=sm_70',
            #                              *nvcc_includes,
            #                              *nvcc_libs,
            #                     ],
            # }
))

setup(name='ops_cpp',
      description='PyTorch Extension for providing fine-structure sparsity accelerated matmul with cuSPARSELt',
      ext_modules=ext_modules,
      cmdclass={'build_ext': cpp_extension.BuildExtension})
