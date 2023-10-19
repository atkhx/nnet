#include "mtl_custom_kernels.h"

// CustomKernelFill

void* customKernelFillCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferFillImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelFill(void *kernelID, void *commandBufferID, void *bufferID, float value) {
    [(__bridge KernelMTLBufferFillImpl*)kernelID
        fill:(id<MTLBuffer>)bufferID withValue:value
        commandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

void customKernelFillPart(void *kernelID, void *commandBufferID, void *bufferID, const uint offset, const uint length, float value) {
    [(__bridge KernelMTLBufferFillImpl*)kernelID
        fillPart:(id<MTLBuffer>)bufferID withValue:value
        commandBuffer:(id<MTLCommandBuffer>)commandBufferID
        offset:offset
        length:length];
}

// CustomKernelReLUFwd

void* customKernelReLUFwdCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferReluFwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelReLUFwd(void *kernelID, void *commandBufferID, void *destinationBufferID, void *sourceBufferID) {
    [(__bridge KernelMTLBufferReluFwdImpl*)kernelID
        reluFwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// CustomKernelReLUBwd

void* customKernelReLUBwdCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferReluBwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelReLUBwd(void *kernelID, void *commandBufferID, void *destinationBufferID, void *sourceBufferID, void *maskBufferID) {
    [(__bridge KernelMTLBufferReluBwdImpl*)kernelID
        reluBwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        maskBuffer:(id<MTLBuffer>)maskBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// CustomKernelMull

void* customKernelMulCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferMulImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelMul(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID
) {
    [(__bridge KernelMTLBufferMulImpl*)kernelID
        mul:(id<MTLBuffer>)destinationBufferID
        multiplierBuffer:(id<MTLBuffer>)multiplierBufferID
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// CustomKernelDropout

void* customKernelDropoutCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferDropoutImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelDropout(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability
) {
    [(__bridge KernelMTLBufferDropoutImpl*)kernelID
        dropout:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        maskOutBuffer:(id<MTLBuffer>)maskOutBufferID
        probability:probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// CustomKernelSoftmax

void* customKernelSoftmaxCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferSoftmaxImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelSoftmax(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge KernelMTLBufferSoftmaxImpl*)kernelID
        softmax:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        sumOutBuffer:(id<MTLBuffer>)sumOutBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// customKernelSoftmaxFwdTrilFwd

void* customKernelSoftmaxTrilFwdCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferSoftmaxTrilImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelSoftmaxTrilFwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge KernelMTLBufferSoftmaxTrilImpl*)kernelID
        softmaxTril:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

// customKernelSoftmaxFwdTrilBwd

void* customKernelSoftmaxTrilBwdCreate(void *deviceID, const char *kernelSource) {
    return [[KernelMTLBufferSoftmaxTrilBwdImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void customKernelSoftmaxTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *softmaxBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
) {
    [(__bridge KernelMTLBufferSoftmaxTrilBwdImpl*)kernelID
        softmaxTrilBwd:(id<MTLBuffer>)destinationBufferID
        sourceBuffer:(id<MTLBuffer>)sourceBufferID
        softmaxBuffer:(id<MTLBuffer>)softmaxBufferID
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}