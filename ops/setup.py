from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='ops_cpp',
      ext_modules=[cpp_extension.CppExtension('ops_cpp', ['ops_cuda.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
