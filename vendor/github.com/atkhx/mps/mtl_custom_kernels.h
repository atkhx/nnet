#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

#include "krn_mtl_buffer_fill.h"
#include "krn_mtl_buffer_relu_fwd.h"
#include "krn_mtl_buffer_relu_bwd.h"
#include "krn_mtl_buffer_mul.h"
#include "krn_mtl_buffer_dropout.h"
#include "krn_mtl_buffer_softmax.h"
#include "krn_mtl_buffer_softmax_tril.h"
#include "krn_mtl_buffer_softmax_tril_bwd.h"

// CustomKernelFill

void* customKernelFillCreate(void *deviceID, const char *kernelSource);
void customKernelFill(void *kernelID, void *commandBufferID, void *bufferID, float value);
void customKernelFillPart(void *kernelID, void *commandBufferID, void *bufferID, const uint offset, const uint length, float value);

// CustomKernelReLUFwd

void* customKernelReLUFwdCreate(void *deviceID, const char *kernelSource);
void customKernelReLUFwd(void *kernelID, void *commandBufferID, void *destinationBufferID, void *sourceBufferID);

// CustomKernelReLUBwd

void* customKernelReLUBwdCreate(void *deviceID, const char *kernelSource);
void customKernelReLUBwd(void *kernelID, void *commandBufferID, void *destinationBufferID, void *sourceBufferID, void *maskBufferID);

// CustomKernelMul

void* customKernelMulCreate(void *deviceID, const char *kernelSource);
void customKernelMul(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *multiplierBufferID
);

// CustomKernelDropout

void* customKernelDropoutCreate(void *deviceID, const char *kernelSource);
void customKernelDropout(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *maskOutBufferID,
    float probability
);

// CustomKernelSoftmax

void* customKernelSoftmaxCreate(void *deviceID, const char *kernelSource);
void customKernelSoftmax(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *sumOutBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

// customKernelSoftmaxFwdTril

void* customKernelSoftmaxTrilFwdCreate(void *deviceID, const char *kernelSource);
void customKernelSoftmaxTrilFwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);

// customKernelSoftmaxBwdTril

void* customKernelSoftmaxTrilBwdCreate(void *deviceID, const char *kernelSource);
void customKernelSoftmaxTrilBwd(
    void *kernelID,
    void *commandBufferID,
    void *destinationBufferID,
    void *sourceBufferID,
    void *softmaxBufferID,
    uint colsCount,
    uint rowsCount,
    uint offset
);
