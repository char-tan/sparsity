from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ops_cpp',
      ext_modules=[cpp_extension.CppExtension('ops_cpp', ['ops_cuda.cpp', 'ops_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
