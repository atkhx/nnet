#import "krn_mtl_buffer_softmax_tril.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferSoftmaxTrilImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionSoftmaxTril;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionSoftmaxTril = [self.library newFunctionWithName:@"softmaxTril"];
        if (!_kernelFunctionSoftmaxTril) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function softmaxTril: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionSoftmaxTril error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) softmaxTril:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTril = [commandBuffer computeCommandEncoder];
    [computeEncoderSoftmaxTril setComputePipelineState:_mFunctionPSO];
    [computeEncoderSoftmaxTril setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTril setBuffer:destinationBuffer offset:offset atIndex:1];
    [computeEncoderSoftmaxTril setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoderSoftmaxTril dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTril endEncoding];
}

@end