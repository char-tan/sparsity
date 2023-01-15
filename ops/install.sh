# /bin/bash

# Download and setup the cuSPARSELT library

# pip3 uninstall cupy-cuda11x

# wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse-lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
# tar -xf libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
# cp -T libcusparse_lt-linux-x86_64-0.3.0.3-archive /content/libcusparse_lt-linux-x86_64-0.3.0.3-archive 
cp /content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib/libcusparseLt* /usr/local/cuda-11.2/targets/x86_64-linux/lib
cp /content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib/libcusparseLt.so /usr/local/cuda-11.2/targets/x86_64-linux/lib/stubs
cp /content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/include/cusparseLt.h /usr/local/cuda-11.2/include

export CUSPARSELT_DIR=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive
export CUSPARSELT_PATH=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive
export LD_LIBRARY_PATH=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib:/usr/lib64-nvidia # :/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib/
# export CUDA_TOOLKIT=$(dirname $(realpath $(which nvcc))/..)
export CUDA_TOOLKIT=/usr/local/cuda-11.2 # HACK to replace line above
echo "Important variables:"
echo $CUSPARSELT_DIR
echo $CUSPARSELT_PATH
echo $LD_LIBRARY_PATH
echo $CUDA_TOOLKIT
ldconfig

CUDA_PATH=/usr/local/cuda-11.2
# pip3 install cupy-cuda112


# Install our PyTorch extension
python setup.py install
pip3 install .
