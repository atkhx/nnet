#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation posEmbeddingsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _posEmbeddingsPSO;
    id<MTLComputePipelineState> _posEmbeddingsGradsPSO;

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

        _posEmbeddingsPSO      = [self createPipelineStateWithFunctionName:@"posEmbeddings"];
        _posEmbeddingsGradsPSO = [self createPipelineStateWithFunctionName:@"posEmbeddingsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        posEmbedding:(id<MTLBuffer>)posEmbedding
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
{
    uint rowsCount = outputData.length/(sizeof(float)*featuresCount);

    id<MTLComputeCommandEncoder> posEmbeddings = [commandBuffer computeCommandEncoder];

    [posEmbeddings setComputePipelineState:_posEmbeddingsPSO];
    [posEmbeddings setBuffer:inputData offset:0 atIndex:0];
    [posEmbeddings setBuffer:outputData offset:0 atIndex:1];
    [posEmbeddings setBuffer:posEmbedding offset:0 atIndex:2];
    [posEmbeddings setBuffer:tokenEmbedding offset:0 atIndex:3];
    [posEmbeddings setBytes:&featuresCount length:sizeof(uint) atIndex:4];
    [posEmbeddings setBytes:&contextLength length:sizeof(uint) atIndex:5];
    [posEmbeddings dispatchThreads:MTLSizeMake(featuresCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [posEmbeddings endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount
{
    uint rowsCount = outputGrad.length/(sizeof(float)*featuresCount);

    id<MTLComputeCommandEncoder> posEmbeddingsGrads = [commandBuffer computeCommandEncoder];
    [posEmbeddingsGrads setComputePipelineState:_posEmbeddingsGradsPSO];
    [posEmbeddingsGrads setBuffer:inputData offset:0 atIndex:0];
    [posEmbeddingsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [posEmbeddingsGrads setBuffer:tokenEmbeddingGrad offset:0 atIndex:2];
    [posEmbeddingsGrads setBytes:&featuresCount length:sizeof(uint) atIndex:3];
    [posEmbeddingsGrads dispatchThreads:MTLSizeMake(featuresCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [posEmbeddingsGrads endEncoding];
}

@end