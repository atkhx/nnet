#include <CoreGraphics/CoreGraphics.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* createDevice();
void releaseDevice(void *deviceID);

void* createNewBufferWithBytes(void *deviceID, float *bytes, size_t length);
void* getBufferContents(void *bufferID);
void releaseBuffer(void *bufferID);

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
);






