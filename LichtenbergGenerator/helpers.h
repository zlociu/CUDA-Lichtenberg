#pragma once

#define HANDLE_ERROR( err ) do { if ((err) != cudaSuccess) { \
    return LICHTENBERG_STATUS_CUDA_ERROR;}} while(0)


#define CHECK(x) do { if((x)!=NVGRAPH_STATUS_SUCCESS) { \
    return LICHTENBERG_STATUS_NV_GRAPH_ERROR;}} while(0)


#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    return LICHTENBERG_STATUS_CURAND_ERROR;}} while(0)