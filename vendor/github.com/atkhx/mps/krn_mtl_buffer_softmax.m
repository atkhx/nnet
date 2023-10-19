#import "krn_mtl_buffer_softmax.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation KernelMTLBufferSoftmaxImpl {
    id<MTLDevice> _device;
    id<MTLFunction> _kernelFunctionExp;
    id<MTLFunction> _kernelFunctionSum;
    id<MTLFunction> _kernelFunctionDiv;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        NSError *error = nil;
        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _kernelFunctionExp = [self.library newFunctionWithName:@"exp"];
        if (!_kernelFunctionExp) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function exp: %s!\n", errorCString);
        }

        _kernelFunctionSum = [self.library newFunctionWithName:@"sum"];
        if (!_kernelFunctionSum) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function sum: %s!\n", errorCString);
        }

        _kernelFunctionDiv = [self.library newFunctionWithName:@"div"];
        if (!_kernelFunctionDiv) {
            const char *errorCString = [[error localizedDescription] UTF8String];
            printf("Failed to load function div: %s!\n", errorCString);
        }
    }
    return self;
}

struct Parameters {
    uint width;
};

- (void) softmax:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
        sumOutBuffer:(id<MTLBuffer>)sumOutBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    NSError *error = nil;

    uint destinationLength = colsCount * rowsCount * 4;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionExp error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState exp: %s\n", errorCString);
    }

    [computeEncoder setBuffer:destinationBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:sourceBuffer offset:offset atIndex:1];

    [computeEncoder dispatchThreads:MTLSizeMake(destinationLength / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];





    id<MTLComputeCommandEncoder> computeEncoderSum = [commandBuffer computeCommandEncoder];
    [computeEncoderSum setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionSum error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState exp: %s\n", errorCString);
    }

    [computeEncoderSum setBuffer:destinationBuffer offset:offset atIndex:0];
    [computeEncoderSum setBuffer:sumOutBuffer offset:0 atIndex:1];

    struct Parameters params;
    params.width = colsCount;
    [computeEncoder setBytes:&params length:sizeof(struct Parameters) atIndex:2];

    [computeEncoderSum dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoderSum endEncoding];


    id<MTLComputeCommandEncoder> computeEncoderDiv = [commandBuffer computeCommandEncoder];
    [computeEncoderDiv setComputePipelineState:[_device newComputePipelineStateWithFunction:_kernelFunctionDiv error:&error]];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to setComputePipelineState exp: %s\n", errorCString);
    }

    [computeEncoderDiv setBuffer:destinationBuffer offset:offset atIndex:0];
    [computeEncoderDiv setBuffer:sumOutBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&params length:sizeof(struct Parameters) atIndex:2];
    [computeEncoderDiv dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoderDiv endEncoding];
}

@end