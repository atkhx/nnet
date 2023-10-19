#import "krn_mtl_buffer_relu_fwd.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferReluFwdImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _reluFunction;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _reluFunction = [self.library newFunctionWithName:@"reluFwd"];
        if (!_reluFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_reluFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void)reluFwd:(id<MTLBuffer>)destinationBuffer sourceBuffer:(id<MTLBuffer>)sourceBuffer withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:destinationBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:sourceBuffer offset:0 atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(destinationBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end