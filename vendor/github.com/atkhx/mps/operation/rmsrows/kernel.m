#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation RmsRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _rmsByRowsPSO;
    id<MTLComputePipelineState> _divRowsDividerGradsPSO;
    
    id<MTLComputePipelineState> _divRowsPSO;
    id<MTLComputePipelineState> _divRowsInputGradsPSO;

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

        _rmsByRowsPSO = [self createPipelineStateWithFunctionName:@"rmsByRows"];

        _divRowsPSO             = [self createPipelineStateWithFunctionName:@"divRows"];
        _divRowsDividerGradsPSO = [self createPipelineStateWithFunctionName:@"divRowsDividerGrads"];
        _divRowsInputGradsPSO   = [self createPipelineStateWithFunctionName:@"divRowsInputGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        aggData:(id<MTLBuffer>)aggData
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> rmsByRows = [commandBuffer computeCommandEncoder];
    [rmsByRows setComputePipelineState:_rmsByRowsPSO];
    [rmsByRows setBuffer:inputData offset:0 atIndex:0];
    [rmsByRows setBuffer:aggData offset:0 atIndex:1];
    [rmsByRows setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [rmsByRows dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [rmsByRows endEncoding];

    id<MTLComputeCommandEncoder> divRowsOnRMS = [commandBuffer computeCommandEncoder];
    [divRowsOnRMS setComputePipelineState:_divRowsPSO];
    [divRowsOnRMS setBuffer:inputData offset:0 atIndex:0];
    [divRowsOnRMS setBuffer:outputData offset:0 atIndex:1];
    [divRowsOnRMS setBuffer:aggData offset:0 atIndex:2];
    [divRowsOnRMS setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [divRowsOnRMS dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [divRowsOnRMS endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> dividerGrads = [commandBuffer computeCommandEncoder];
    [dividerGrads setComputePipelineState:_divRowsDividerGradsPSO];
    [dividerGrads setBuffer:aggData offset:0 atIndex:0];
    [dividerGrads setBuffer:aggGrad offset:0 atIndex:1];
    [dividerGrads setBuffer:outputData offset:0 atIndex:2];
    [dividerGrads setBuffer:outputGrad offset:0 atIndex:3];
    [dividerGrads setBytes:&chunkSize length:sizeof(uint) atIndex:4];
    [dividerGrads dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [dividerGrads endEncoding];

    id<MTLComputeCommandEncoder> inputGrads = [commandBuffer computeCommandEncoder];
    [inputGrads setComputePipelineState:_divRowsInputGradsPSO];
    [inputGrads setBuffer:inputData offset:0 atIndex:0];
    [inputGrads setBuffer:inputGrad offset:0 atIndex:1];
    [inputGrads setBuffer:outputGrad offset:0 atIndex:2];
    [inputGrads setBuffer:aggData offset:0 atIndex:3];
    [inputGrads setBuffer:aggGrad offset:0 atIndex:4];
    [inputGrads setBytes:&chunkSize length:sizeof(uint) atIndex:5];
    [inputGrads dispatchThreads:MTLSizeMake(inputGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [inputGrads endEncoding];
}

@end