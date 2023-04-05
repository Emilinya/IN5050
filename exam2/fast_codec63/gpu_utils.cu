#include <cuda.h>

#include "c63.h"
#include "utils.h"
#include "gpu_utils.h"

__host__ struct gpu_frame *gpu_init(struct c63_common *cm)
{
    struct gpu_frame *frame;
    cudaMallocManaged((void **)&frame, sizeof(struct gpu_frame));

    // allocate memory on the device for the input
    cudaMallocManaged((void **)&frame->input, sizeof(yuv_t));
    cudaMallocManaged((void **)&frame->input->Y, cm->yph * cm->ypw);
    cudaMallocManaged((void **)&frame->input->U, cm->uph * cm->upw);
    cudaMallocManaged((void **)&frame->input->V, cm->vph * cm->upw);
    
    // Allocate memory on the device for the reference
    cudaMallocManaged((void **)&frame->reference, sizeof(yuv_t));
    cudaMallocManaged((void **)&frame->reference->Y, cm->yph * cm->ypw);
    cudaMallocManaged((void **)&frame->reference->U, cm->uph * cm->upw);
    cudaMallocManaged((void **)&frame->reference->V, cm->vph * cm->upw);

    return frame;
}

__host__ void gpu_cleanup(struct gpu_frame *gpu_frame)
{
    cudaFree(gpu_frame->input->Y);
    cudaFree(gpu_frame->input->U);
    cudaFree(gpu_frame->input->V);
    cudaFree(gpu_frame->input);

    cudaFree(gpu_frame->reference->Y);
    cudaFree(gpu_frame->reference->U);
    cudaFree(gpu_frame->reference->V);
    cudaFree(gpu_frame->reference);

    cudaFree(gpu_frame);
}
