#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation ropeRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _ropeRowsPSO;
    id<MTLComputePipelineState> _ropeRowsGradsPSO;

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
    
        _ropeRowsPSO      = [self createPipelineStateWithFunctionName:@"ropeRows"];
        _ropeRowsGradsPSO = [self createPipelineStateWithFunctionName:@"ropeRowsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputData.length/(sizeof(float) * featuresCount * contextLength);
    id<MTLComputeCommandEncoder> ropeRows = [commandBuffer computeCommandEncoder];

    [ropeRows setComputePipelineState:_ropeRowsPSO];
    [ropeRows setBuffer:inputData offset:0 atIndex:0];
    [ropeRows setBuffer:outputData offset:0 atIndex:1];
    [ropeRows setBytes:&featuresCount length:sizeof(uint) atIndex:2];
    [ropeRows setBytes:&headSize length:sizeof(uint) atIndex:3];
    [ropeRows setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [ropeRows dispatchThreads:MTLSizeMake(featuresCount/2, contextLength, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ropeRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputGrad.length/(sizeof(float) * featuresCount * contextLength);
    id<MTLComputeCommandEncoder> ropeRowsGrads = [commandBuffer computeCommandEncoder];

    [ropeRowsGrads setComputePipelineState:_ropeRowsGradsPSO];
    [ropeRowsGrads setBuffer:inputGrad offset:0 atIndex:0];
    [ropeRowsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [ropeRowsGrads setBytes:&featuresCount length:sizeof(uint) atIndex:2];
    [ropeRowsGrads setBytes:&headSize length:sizeof(uint) atIndex:3];
    [ropeRowsGrads setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [ropeRowsGrads dispatchThreads:MTLSizeMake(featuresCount/2, contextLength, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ropeRowsGrads endEncoding];
}

@end