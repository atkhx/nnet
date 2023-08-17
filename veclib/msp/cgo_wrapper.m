// cgo_wrapper.m
#include "cgo_wrapper.h"

void* createDevice() {
    return MTLCreateSystemDefaultDevice();
}

void releaseDevice(void *deviceID) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    [device release];
}

void* createNewBufferWithBytes(void *deviceID, float *bytes, size_t length) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLBuffer> buffer = [device
        newBufferWithBytes:bytes
        length:length * sizeof(float)
        options: MTLResourceCPUCacheModeDefaultCache
    ];
     return (__bridge void*)buffer;
}

void releaseBuffer(void *bufferID) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    [buffer release];
}

void* getBufferContents(void *bufferID) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)bufferID;
    return [buffer contents];
}

void matrixMultiplyOnDevice(
    void *deviceID,

    void *bufferIDA,
    void *bufferIDB,
    void *bufferIDC,

    int aW, int aH,
    int bW, int bH,
    int cW, int cH,

    int _interiorColumns,
    double _alpha, double _beta,
    bool _transposeLeft, bool _transposeRight
) {
    id<MTLDevice> device = (id<MTLDevice>)deviceID;
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:aH columns:aW rowBytes:aW * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:bH columns:bW rowBytes:bW * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:cH columns:cW rowBytes:cW * sizeof(float) dataType:MPSDataTypeFloat32];

    id<MTLBuffer> bufferA = (id<MTLBuffer>)bufferIDA;
    id<MTLBuffer> bufferB = (id<MTLBuffer>)bufferIDB;
    id<MTLBuffer> bufferC = (id<MTLBuffer>)bufferIDC;

    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
        transposeLeft:_transposeLeft
        transposeRight:_transposeRight
        resultRows:cH
        resultColumns:cW
        interiorColumns:_interiorColumns
        alpha:_alpha
        beta:_beta];

    [kernel encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    [matrixA release];
    [matrixB release];
    [matrixC release];

    [descA release];
    [descB release];
    [descC release];

    [kernel release];
    [commandQueue release];
    [commandBuffer release];
}
