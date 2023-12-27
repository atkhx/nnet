#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

// MPSMatrixDescriptor

void* mpsMatrixDescriptorCreate(int cols, int rows, int batchSize, int batchStride);
void mpsMatrixDescriptorRelease(void *descriptorID);

// MPSMatrix

void* mpsMatrixCreate(void *bufferID, void *descriptorID, int offset);
void mpsMatrixRelease(void *matrixID);

// MPSMatrixMultiplication

void* mpsMatrixMultiplicationCreate(
    void *deviceID,

    int resultRows,
    int resultColumns,
    int interiorColumns,

    float alpha,
    float beta,

    bool transposeLeft,
    bool transposeRight
);

void mpsMatrixMultiplicationEncode(
    void *commandBufferID,
    void *kernelID,
    void *matrixAID,
    void *matrixBID,
    void *matrixCID
);

// MPSMatrixRandomDistributionDescriptor

void* mpsMatrixRandomDistributionDescriptorCreate(float min, float max);

// MPSMatrixRandomMTGP32

void* mpsMatrixRandomMTGP32Create(void *deviceID, void *distribution, NSUInteger seed);
void mpsMatrixRandomMTGP32Encode(void *kernelID, void *commandBufferID, void *dstMatrix);

// MPSMatrixSoftMax
void* mpsMatrixSoftMaxCreate(void *deviceID);
void mpsMatrixSoftMaxEncode(
    void *commandBufferID,
    void *kernelID,
    void *inputMatrix,
    void *resultMatrix
);

// MPSMatrixSoftMaxGradient

void* mpsMatrixSoftMaxGradientCreate(void *deviceID);
void mpsMatrixSoftMaxGradientEncode(
    void *commandBufferID,
    void *kernelID,
    void *gradientMatrix,
    void *forwardOutputMatrix,
    void *resultMatrix
);


