#import "krn_mtl_buffer_add.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferAddImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionAdd;
    id<MTLComputePipelineState> _mFunctionAddPSO;
    id<MTLFunction> _kernelFunctionAddTo;
    id<MTLComputePipelineState> _mFunctionAddToPSO;

    NSError *error;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _kernelFunctionAdd = [self.library newFunctionWithName:@"add"];
        if (!_kernelFunctionAdd) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function add: %s!\n", errorCString);
        }

        _mFunctionAddPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionAdd error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction add: %s\n", errorCString);
        }

        _kernelFunctionAddTo = [self.library newFunctionWithName:@"addTo"];
        if (!_kernelFunctionAdd) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function addTo: %s!\n", errorCString);
        }

        _mFunctionAddToPSO = [_device newComputePipelineStateWithFunction:_kernelFunctionAddTo error:&error];
        if (error != nil) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("newComputePipelineStateWithFunction addTo: %s\n", errorCString);
        }
    }
    return self;
}

- (void) add:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionAddPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addTo:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mFunctionAddToPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:aBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:bBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

@end