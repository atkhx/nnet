#import "krn_mtl_buffer_mul.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferMulImpl {
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
        _kernelFunction = [self.library newFunctionWithName:@"mul"];
        if (!_kernelFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function mul: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) mul:(id<MTLBuffer>)destinationBuffer
        multiplierBuffer:(id<MTLBuffer>)multiplierBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:destinationBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:multiplierBuffer offset:0 atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(destinationBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end