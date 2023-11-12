#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

#include "custom_kernel.h"

void* customKernelCreate(void *deviceID, const char *kernelSource);

void customKernelCopy(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
);

void customKernelFill(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value,
    const uint offset,
    const uint length
);

void customKernelAdd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
);
void customKernelAddTo(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *aBuffer,
    void *bBuffer
);

void customKernelAddScalar(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value
);

void customKernelMul(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
);

void customKernelReLU(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID
);
void customKernelReLUBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID
);

void customKernelSoftmax(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

void customKernelDropout(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
);
void customKernelDropoutBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
);

void customKernelUpdateWithAdam(
    void *kernelID,
    void *commandBufferID,

    void *dataBufferID,
    void *gradBufferID,
    void *mBufferID,
    void *vBufferID,

    float beta1,
    float beta2,
    float beta1powIterationLR,
    float beta2powIteration
);

void customKernelSoftmaxTrilFwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

void customKernelSoftmaxTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

void customKernelCrossEntropyPos(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    void *sumBufferID,
    void *tgtBufferID,
    uint chunkSize
);

void customKernelCrossEntropyPosBwd(
    void *kernelID,
    void *commandBufferID,
    void *oGradBufferID,
    void *aGradBufferID,
    void *tgtBufferID,
    void *smxBufferID,
    uint chunkSize
);

void customKernelRMSNorm(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint chunkSize
);

