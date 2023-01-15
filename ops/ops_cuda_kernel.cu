#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparseLt.h> // cusparseLt header


// template <typename scalar_t>
// __global__ void cuda_add_kernel(
//     scalar_t* __restrict__ a,
//     scalar_t* __restrict__ b,
//     scalar_t* __restrict__ c
//     ) {
//         auto c = torch::add(a, b);
// 	    return c;
//       }


template <typename scalar_t>
__global__ void cuda_sparse_mm_kernel(
    scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ c
    ) {

        // Device pointers and coefficient definitions
        float alpha = 1.0f;
        float beta  = 0.0f;
        __half* dA = ...
        __half* dB = ...
        __half* dC = ...

        //--------------------------------------------------------------------------
        // cusparseLt data structures and handle initialization
        cusparseLtHandle_t             handle;
        cusparseLtMatDescriptor_t      matA, matB, matC;
        cusparseLtMatmulDescriptor_t   matmul;
        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseLtMatmulPlan_t         plan;
        cudaStream_t                   stream = nullptr;
        cusparseLtInit(&handle);

        //--------------------------------------------------------------------------
        // matrix descriptor initialization
        cusparseLtStructuredDescriptorInit(&handle, &matA, num_A_rows, num_A_cols,
                                          lda, alignment, type, order,
                                          CUSPARSELT_SPARSITY_50_PERCENT);
        cusparseLtDenseDescriptorInit(&handle, &matB, num_B_rows, num_B_cols, ldb,
                                      alignment, type, order);
        cusparseLtDenseDescriptorInit(&handle, &matC, num_C_rows, num_C_cols, ldc,
                                      alignment, type, order);

        //--------------------------------------------------------------------------
        // matmul, algorithm selection, and plan initialization
        cusparseLtMatmulDescriptorInit(&handle, &matmul, opA, opB, &matA, &matB,
                                      &matC, &matC, compute_type);
        cusparseLtMatmulAlgSelectionInit(&handle, &alg_sel, &matmul,
                                        CUSPARSELT_MATMUL_ALG_DEFAULT);
        cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel, workspace_size);

        //--------------------------------------------------------------------------
        // Prune the A matrix (in-place) and check the correctness
        cusparseLtSpMMAPrune(&handle, &matmul, dA, dA, CUSPARSELT_PRUNE_SPMMA_TILE,
                            stream);
        int *d_valid;
        cudaMalloc((void**) &d_valid, sizeof(d_valid));
        cusparseLtSpMMAPruneCheck(&handle, &matmul, dA, &d_valid, stream);

        int is_valid;
        cudaMemcpyAsync(&is_valid, d_valid, sizeof(d_valid), cudaMemcpyDeviceToHost,
                        stream);
        cudaStreamSynchronize(stream);
        if (is_valid != 0) {
            std::printf("!!!! The matrix has been pruned in a wrong way. "
                        "cusparseLtMatmul will not provided correct results\n");
            return EXIT_FAILURE;
        }

        //--------------------------------------------------------------------------
        // Matrix A compression
        size_t compressed_size;
        cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size);
        cudaMalloc((void**) &dA_compressed, compressed_size);

        cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed, stream);

        //--------------------------------------------------------------------------
        // Allocate workspace
        size_t workspace_size;
        void*  d_workspace = nullptr;

        cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
        cudaMalloc((void**) &d_workspace, workspace_size);

        //--------------------------------------------------------------------------
        // Perform the matrix multiplication
        int           num_streams = 0;
        cudaStream_t* streams     = nullptr;

        cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB, &beta, dC, dD,
                        d_workspace, streams, num_streams)

        //--------------------------------------------------------------------------
        // Destroy descriptors, plan and handle
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtDestroy(&handle);

        c = a
      }


torch::Tensor cuda_add(
		torch::Tensor a,
		torch::Tensor b)
{

    auto c = torch::add(a, b);
    return c;
}

torch::Tensor cuda_sparse_mm(
		torch::Tensor a,
		torch::Tensor b)
{

  // auto c = torch::zeros_like(a);
    auto c = torch::add(a, b);
    return c;
}
