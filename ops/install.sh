# /bin/bash

# Download and setup the cuSPARSELT library

wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse-lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
tar -xf libcusparse_lt-linux-x86_64-0.3.0.3-archive.tar.xz
mv libcusparse_lt-linux-x86_64-0.3.0.3-archive /content
export CUSPARSELT_DIR=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive
export LD_LIBRARY_PATH=${CUSPARSELT_DIR}/lib64:${LD_LIBRARY_PATH}

# 
export CUSPARSELT_DIR=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive
export CUSPARSELT_PATH=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive
export LD_LIBRARY_PATH=/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib64:/usr/lib64-nvidia:/content/libcusparse_lt-linux-x86_64-0.3.0.3-archive/lib/
export CUDA_TOOLKIT=$(dirname $(realpath $(which nvcc))/..)
echo "Important variables:"
echo $CUSPARSELT_DIR
echo $CUSPARSELT_PATH
echo $LD_LIBRARY_PATH
echo $CUDA_TOOLKIT
ldconfig

# Install our PyTorch extension
!pip3 install .
