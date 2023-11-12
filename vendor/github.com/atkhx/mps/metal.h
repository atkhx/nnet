#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

// MTLDevice

void* mtlDeviceCreate();
void mtlDeviceRelease(void *deviceID);

// MTLCommandQueue

void* mtlCommandQueueCreate(void *deviceID);
void mtlCommandQueueRelease(void *commandQueueID);

// MTLCommandBuffer

void* mtlCommandBufferCreate(void *commandQueueID);
void mtlCommandBufferRelease(void *commandBufferID);
void mtlCommandBufferCommitAndWaitUntilCompleted(void *commandBufferID);

// MTLBuffer

void* mtlBufferCreateCreateWithBytes(void *deviceID, float *bytes, size_t length);
void* mtlBufferCreateWithLength(void *deviceID, size_t length);
void* mtlBufferGetContents(void *bufferID);
void mtlBufferRelease(void *bufferID);

// MPSMatrixDescriptor

void* mpsMatrixDescriptorCreate(int cols, int rows);
void mpsMatrixDescriptorRelease(void *descriptorID);

// MPSMatrix

void* mpsMatrixCreate(void *bufferID, void *descriptorID, int offset);
void mpsMatrixRelease(void *matrixID);

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
);

// Random
void* mpsMatrixRandomDistributionCreate(float min, float max);
void* mpsMatrixRandomMTGP32Create(void *deviceID, void *distribution, NSUInteger seed);
void mpsMatrixRandom(void *kernelID, void *commandBufferID, void *dstMatrix);

