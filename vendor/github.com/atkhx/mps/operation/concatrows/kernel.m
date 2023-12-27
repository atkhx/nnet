#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation ConcatRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _concatRowsPSO;
    id<MTLComputePipelineState> _concatRowsBwdPSO;

    NSError *error;
}

- (id<MTLComputePipelineState>)createPipelineStateWithFunctionName:(NSString *)functionName {
    id<MTLFunction> function = [self.library newFunctionWithName:functionName];
    if (!function) {
        printf("Failed to load function %s!\n", [functionName UTF8String]);
        return nil;
    }

    id<MTLComputePipelineState> pipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to create pipeline state: %s\n", errorCString);
        return nil;
    }
    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _concatRowsPSO = [self createPipelineStateWithFunctionName:@"concatRows"];
        _concatRowsBwdPSO = [self createPipelineStateWithFunctionName:@"concatRowsBwd"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
{
    uint rowsCount = inputData.length/(sizeof(float)*inputWidth);

    id<MTLComputeCommandEncoder> concatRows = [commandBuffer computeCommandEncoder];
    [concatRows setComputePipelineState:_concatRowsPSO];
    [concatRows setBuffer:inputData offset:0 atIndex:0];
    [concatRows setBuffer:outputData offset:0 atIndex:1];
    [concatRows setBytes:&inputWidth length:sizeof(uint) atIndex:2];
    [concatRows setBytes:&outputWidth length:sizeof(uint) atIndex:3];
    [concatRows setBytes:&outputOffset length:sizeof(uint) atIndex:4];
    [concatRows dispatchThreads:MTLSizeMake(inputWidth, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [concatRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inputWidth:(uint)inputWidth
        outputWidth:(uint)outputWidth
        outputOffset:(uint)outputOffset
{
    uint rowsCount = inputGrad.length/(sizeof(float)*inputWidth);

    id<MTLComputeCommandEncoder> concatRowsBwd = [commandBuffer computeCommandEncoder];
    [concatRowsBwd setComputePipelineState:_concatRowsBwdPSO];
    [concatRowsBwd setBuffer:inputGrad offset:0 atIndex:0];
    [concatRowsBwd setBuffer:outputGrad offset:0 atIndex:1];
    [concatRowsBwd setBytes:&inputWidth length:sizeof(uint) atIndex:2];
    [concatRowsBwd setBytes:&outputWidth length:sizeof(uint) atIndex:3];
    [concatRowsBwd setBytes:&outputOffset length:sizeof(uint) atIndex:4];
    [concatRowsBwd dispatchThreads:MTLSizeMake(inputWidth, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [concatRowsBwd endEncoding];
}

@end