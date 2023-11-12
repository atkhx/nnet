#include "mtl_custom_kernel.h"

void* customKernelCreate(void *deviceID, const char *kernelSource) {
    return [[MPSCustomKernelImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelCopy(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID copy:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}

void customKernelFill(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value,
    const uint offset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID fill:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        value:value
        offset:offset
        length:length];
}

void customKernelAdd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID add:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}

void customKernelAddTo(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *aBuffer,
    void *bBuffer
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addTo:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer];
}

void customKernelAddScalar(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    float value
) {
    [(__bridge MPSCustomKernelImpl*)kernelID addScalar:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        value:value];
}

void customKernelMul(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    const uint dstOffset,
    const uint srcOffset,
    const uint length
) {
    [(__bridge MPSCustomKernelImpl*)kernelID mul:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        dstOffset:dstOffset
        srcOffset:srcOffset
        length:length];
}

void customKernelReLU(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID
) {
    [(__bridge MPSCustomKernelImpl*)kernelID relu:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID];
}

void customKernelReLUBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID
) {
    [(__bridge MPSCustomKernelImpl*)kernelID reluBwd:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID];
}

void customKernelSoftmax(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmax:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelDropout(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
) {
    [(__bridge MPSCustomKernelImpl*)kernelID dropout:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID
        probability:probability];
}

void customKernelDropoutBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *mskBufferID,
    float probability
) {
    [(__bridge MPSCustomKernelImpl*)kernelID
        dropoutBwd:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        mskBuffer:(id<MTLBuffer>)mskBufferID
        probability:probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

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
) {
    [(__bridge MPSCustomKernelImpl*)kernelID updateWithAdam:(id<MTLCommandBuffer>)commandBufferID
        dataBuffer:(id<MTLBuffer>)dataBufferID
        gradBuffer:(id<MTLBuffer>)gradBufferID
        mBuffer:(id<MTLBuffer>)mBufferID
        vBuffer:(id<MTLBuffer>)vBufferID
        beta1:beta1
        beta2:beta2
        beta1powIterationLR:beta1powIterationLR
        beta2powIteration:beta2powIteration];
}

void customKernelSoftmaxTrilFwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmaxTril:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelSoftmaxTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge MPSCustomKernelImpl*)kernelID softmaxTrilBwd:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];
}

void customKernelCrossEntropyPos(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *smxBufferID,
    void *sumBufferID,
    void *tgtBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID crossEntropyPos:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        tgtBuffer:(id<MTLBuffer>)tgtBufferID
        chunkSize:chunkSize];
}

void customKernelCrossEntropyPosBwd(
    void *kernelID,
    void *commandBufferID,
    void *oGradBufferID,
    void *aGradBufferID,
    void *tgtBufferID,
    void *smxBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBufferID
        oGrad:(id<MTLBuffer>)oGradBufferID
        aGrad:(id<MTLBuffer>)aGradBufferID
        tgtBuffer:(id<MTLBuffer>)tgtBufferID
        smxBuffer:(id<MTLBuffer>)smxBufferID
        chunkSize:chunkSize];
}

void customKernelRMSNorm(
    void *kernelID,
    void *commandBufferID,
    void *dstBufferID,
    void *srcBufferID,
    void *sumBufferID,
    uint chunkSize
) {
    [(__bridge MPSCustomKernelImpl*)kernelID rmsNorm:(id<MTLCommandBuffer>)commandBufferID
        dstBuffer:(id<MTLBuffer>)dstBufferID
        srcBuffer:(id<MTLBuffer>)srcBufferID
        sumBuffer:(id<MTLBuffer>)sumBufferID
        chunkSize:chunkSize];
}

