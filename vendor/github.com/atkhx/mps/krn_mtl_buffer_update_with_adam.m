#import "krn_mtl_buffer_update_with_adam.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferUpdateWithAdamImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionUpdateWithAdam;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionUpdateWithAdam = [self.library newFunctionWithName:@"updateWithAdam"];
        if (!_kernelFunctionUpdateWithAdam) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function updateWithAdam: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionUpdateWithAdam error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) updateWithAdam:(id<MTLBuffer>)dataBuffer
            gradBuffer:(id<MTLBuffer>)gradBuffer
            mBuffer:(id<MTLBuffer>)mBuffer
            vBuffer:(id<MTLBuffer>)vBuffer
            beta1:(float)beta1
            beta2:(float)beta2
            beta1powIterationLR:(float)beta1powIterationLR
            beta2powIteration:(float)beta2powIteration
            withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:dataBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:gradBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mBuffer    offset:0 atIndex:2];
    [computeEncoder setBuffer:vBuffer    offset:0 atIndex:3];
    [computeEncoder setBytes:&beta1 length:sizeof(float) atIndex:4];
    [computeEncoder setBytes:&beta2 length:sizeof(float) atIndex:5];
    [computeEncoder setBytes:&beta1powIterationLR length:sizeof(float) atIndex:6];
    [computeEncoder setBytes:&beta2powIteration length:sizeof(float) atIndex:7];
    [computeEncoder dispatchThreads:MTLSizeMake(dataBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end