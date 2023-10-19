#import "krn_mtl_buffer_dropout.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferDropoutImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunction;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _kernelFunction = [self.library newFunctionWithName:@"dropout"];
        if (!_kernelFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function dropout: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) dropout:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        maskOutBuffer:(id<MTLBuffer>)maskOutBuffer
        probability:(float)probability
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:destinationBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:sourceBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:maskOutBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&probability length:sizeof(float) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(destinationBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end