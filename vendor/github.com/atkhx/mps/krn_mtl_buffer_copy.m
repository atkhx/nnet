#import "krn_mtl_buffer_copy.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferCopyImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunction;
    id<MTLComputePipelineState> _mFunctionPSO;
    NSError *error;
}

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _kernelFunction = [self.library newFunctionWithName:@"copy"];

        if (!_kernelFunction) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function: %s!\n", errorCString);
        }

        _mFunctionPSO = [_device newComputePipelineStateWithFunction:_kernelFunction error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction: %s\n", errorCString);
        }
    }
    return self;
}

- (void) copy:(id<MTLBuffer>)dstBuffer
    srcBuffer:(id<MTLBuffer>)srcBuffer
    dstOffset:(uint)dstOffset
    srcOffset:(uint)srcOffset
    length:(uint)length
    withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
 {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end