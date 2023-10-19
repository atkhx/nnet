#import "krn_mtl_buffer_softmax_tril_bwd.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferSoftmaxTrilBwdImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionSoftmaxTrilBwd;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionSoftmaxTrilBwd = [self.library newFunctionWithName:@"softmaxTrilBwd"];
        if (!_kernelFunctionSoftmaxTrilBwd) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function softmaxTrilBwd: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionSoftmaxTrilBwd error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) softmaxTrilBwd:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        softmaxBuffer:(id<MTLBuffer>)softmaxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoderSoftmaxTrilBwd = [commandBuffer computeCommandEncoder];

    [computeEncoderSoftmaxTrilBwd setComputePipelineState:_mFunctionPSO];
    [computeEncoderSoftmaxTrilBwd setBuffer:sourceBuffer offset:offset atIndex:0];
    [computeEncoderSoftmaxTrilBwd setBuffer:destinationBuffer offset:offset atIndex:1];
    [computeEncoderSoftmaxTrilBwd setBuffer:softmaxBuffer offset:offset atIndex:2];
    [computeEncoderSoftmaxTrilBwd setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoderSoftmaxTrilBwd dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSoftmaxTrilBwd endEncoding];
}

@end