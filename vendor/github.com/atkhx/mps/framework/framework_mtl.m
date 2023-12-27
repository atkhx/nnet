#include "framework_mtl.h"

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
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull cmdBuf) {
        NSError *err = cmdBuf.error;
        if(err){
            NSLog(@"%@", err);
        }
    }];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

// MTLBuffer

void* mtlBufferCreateWithBytes(void *deviceID, float *bytes, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithBytes:bytes
        length:length*sizeof(float)
        options:MTLResourceStorageModeShared];
}

void* mtlBufferCreateWithLength(void *deviceID, size_t length) {
    return [(id<MTLDevice>)deviceID
        newBufferWithLength:length * sizeof(float)
        options:MTLResourceStorageModeShared];
}

void* mtlBufferGetContents(void *bufferID) {
    return [(id<MTLBuffer>)bufferID contents];
}

void mtlBufferRelease(void *bufferID) {
    [(id<MTLBuffer>)bufferID release];
}

void mtlCopyToBuffer(
    void *deviceID,
    void *srcBufferID,
    void *dstBufferID,
    size_t bufferSize
) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceID;
    id<MTLBuffer> srcBuffer = (__bridge id<MTLBuffer>)srcBufferID;
    id<MTLBuffer> dstBuffer = (__bridge id<MTLBuffer>)dstBufferID;

    // Создание командной очереди
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Создание буфера для хранения командной последовательности
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Создание командного энкодера для записи команд
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

    // Копирование данных из приватного буфера в публичный
    [blitEncoder copyFromBuffer:srcBuffer
                  sourceOffset:0
                      toBuffer:dstBuffer
             destinationOffset:0
                          size:bufferSize*sizeof(float)];

    // Завершение кодирования команд
    [blitEncoder endEncoding];

    // Выполнение команд
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    [blitEncoder release];
    [commandBuffer release];
    [commandQueue release];
}


void mtlCopyToBufferWithCommandBuffer(
    void *commandBufferID,
    void *srcBufferID,
    void *dstBufferID,
    size_t bufferSize
) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)commandBufferID;

    // Создание командного энкодера для записи команд
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

    // Копирование данных из приватного буфера в публичный
    [blitEncoder copyFromBuffer:(__bridge id<MTLBuffer>)srcBufferID
                  sourceOffset:0
                      toBuffer:(__bridge id<MTLBuffer>)dstBufferID
             destinationOffset:0
                          size:bufferSize*sizeof(float)];

    // Завершение кодирования команд
    [blitEncoder endEncoding];
}