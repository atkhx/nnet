#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

#include "kernel.h"

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

void customKernelCopyWHD(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint W,
    const uint H,
    const uint D
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
void customKernelAddToWHD(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *aBuffer,
    void *bBuffer,
    const float K,
    const uint W,
    const uint H,
    const uint D
);
void customKernelAddToWHDBwd(
    void *kernelID,
    void *commandBufferID,
    void *aGrad,
    void *bGrad,
    void *oGrad,
    const uint W,
    const uint H,
    const uint D
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

void customKernelNLLByPos(
    void *kernelID,
    void *commandBufferID,
    void *dstBuffer,
    void *smxBuffer,
    void *tgtBuffer,
    uint chunkSize
);

void customKernelNLLByPosBwd(
    void *kernelID,
    void *commandBufferID,
    void *oGradBufferID,
    void *aGradBufferID,
    void *tgtBufferID,
    void *smxBufferID,
    uint chunkSize);

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

void transposeTo(
    void *kernelID,
    void *commandBufferID,
    void *inputData,
    void *outputData,
    uint width,
    uint height
);

void transposeAndAddTo(
    void *kernelID,
    void *commandBufferID,
    void *inputData,
    void *outputData,
    uint width,
    uint height
);
