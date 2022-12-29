from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sparse_ops_cpp',
      ext_modules=[cpp_extension.CppExtension('sparse_ops_cpp', ['test.cpp', 'sparse_ops.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
