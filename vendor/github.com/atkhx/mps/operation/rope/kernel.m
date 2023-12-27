#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation RopeKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _ropePSO;
    id<MTLComputePipelineState> _ropeGradsPSO;

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

        _ropePSO      = [self createPipelineStateWithFunctionName:@"rope"];
        _ropeGradsPSO = [self createPipelineStateWithFunctionName:@"ropeGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputData.length/(sizeof(float) * headSize * contextLength);
    id<MTLComputeCommandEncoder> rope = [commandBuffer computeCommandEncoder];

    [rope setComputePipelineState:_ropePSO];
    [rope setBuffer:inputData offset:0 atIndex:0];
    [rope setBuffer:outputData offset:0 atIndex:1];
    [rope setBytes:&headIndex length:sizeof(uint) atIndex:2];
    [rope setBytes:&headSize length:sizeof(uint) atIndex:3];
    [rope setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [rope dispatchThreads:MTLSizeMake(headSize/2, contextLength, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [rope endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        headIndex:(uint)headIndex
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputGrad.length/(sizeof(float) * headSize * contextLength);
    id<MTLComputeCommandEncoder> ropeGrads = [commandBuffer computeCommandEncoder];

    [ropeGrads setComputePipelineState:_ropeGradsPSO];
    [ropeGrads setBuffer:inputGrad offset:0 atIndex:0];
    [ropeGrads setBuffer:outputGrad offset:0 atIndex:1];
    [ropeGrads setBytes:&headIndex length:sizeof(uint) atIndex:2];
    [ropeGrads setBytes:&headSize length:sizeof(uint) atIndex:3];
    [ropeGrads setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [ropeGrads dispatchThreads:MTLSizeMake(headSize/2, contextLength, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ropeGrads endEncoding];
}

@end