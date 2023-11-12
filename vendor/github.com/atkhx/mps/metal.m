#include "metal.h"

// MTLDevice

void* mtlDeviceCreate() {
    return MTLCreateSystemDefaultDevice();
}

void mtlDeviceRelease(void *deviceID) {
    [(id<MTLDevice>)deviceID release];
}

// MTLCommandQueue

void* mtlCommandQueueCreate(void *deviceID) {
    return [(id<MTLDevice>)deviceID newCommandQueue];
}

void mtlCommandQueueRelease(void *commandQueueID) {
    [(id<MTLCommandQueue>)commandQueueID release];
}

// MTLCommandBuffer

void* mtlCommandBufferCreate(void *commandQueueID) {
    return [(id<MTLCommandQueue>)commandQueueID commandBuffer];
}

void mtlCommandBufferRelease(void *commandBufferID) {
    [(id<MTLCommandBuffer>)commandBufferID release];
}

void mtlCommandBufferCommitAndWaitUntilCompleted(void *commandBufferID) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// MTLBuffer

void* mtlBufferCreateCreateWithBytes(void *deviceID, float *bytes, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithBytes:bytes
        length:length*sizeof(float)
        options: MTLResourceStorageModeShared];
}

void* mtlBufferCreateWithLength(void *deviceID, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithLength:length * sizeof(float)
        options: MTLResourceStorageModeShared];
}

void* mtlBufferGetContents(void *bufferID) {
    return [(id<MTLBuffer>)bufferID contents];
}

void mtlBufferRelease(void *bufferID) {
    [(id<MTLBuffer>)bufferID release];
}

// MPSMatrixDescriptor

void* mpsMatrixDescriptorCreate(int cols, int rows) {
    return [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
        columns:cols
        rowBytes:cols * sizeof(float)
        dataType:MPSDataTypeFloat32];
}

void mpsMatrixDescriptorRelease(void *descriptorID) {
    [(__bridge MPSMatrixDescriptor*)descriptorID release];
}

// MPSMatrix

void* mpsMatrixCreate(void *bufferID, void *descriptorID, int offset) {
    return [[MPSMatrix alloc]
        initWithBuffer:(id<MTLBuffer>)bufferID
        offset:offset*sizeof(float)
        descriptor:(__bridge MPSMatrixDescriptor*)descriptorID];
}

void mpsMatrixRelease(void *matrixID) {
    [(__bridge MPSMatrix*)matrixID release];
}

void mpsMatrixMultiply(
    void *deviceID,
    void *commandBufferID,

    void *matrixAID,
    void *matrixBID,
    void *matrixCID,

    int _interiorColumns,

    float _alpha,
    float _beta,

    bool _transposeLeft,
    bool _transposeRight
) {
    MPSMatrix *matrixC = (__bridge MPSMatrix*)matrixCID;

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        transposeLeft:_transposeLeft
        transposeRight:_transposeRight
        resultRows:matrixC.rows
        resultColumns:matrixC.columns
        interiorColumns:_interiorColumns
        alpha:_alpha
        beta:_beta];

    [kernel
        encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        leftMatrix:(__bridge MPSMatrix*)matrixAID
        rightMatrix:(__bridge MPSMatrix*)matrixBID
        resultMatrix:matrixC];

    [kernel release];
}

void* mpsMatrixRandomDistributionCreate(float min, float max) {
    return [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:min maximum:max];
}

void* mpsMatrixRandomMTGP32Create(void *deviceID, void *distribution, NSUInteger seed) {
    return [[MPSMatrixRandomMTGP32 alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        destinationDataType:MPSDataTypeFloat32
        seed:seed
        distributionDescriptor:(__bridge MPSMatrixRandomDistributionDescriptor*)distribution
    ];
}

void mpsMatrixRandom(void *kernelID, void *commandBufferID, void *dstMatrix) {
    [(__bridge MPSMatrixRandomMTGP32*)kernelID
        encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        destinationMatrix:(__bridge MPSMatrix*)dstMatrix];
}