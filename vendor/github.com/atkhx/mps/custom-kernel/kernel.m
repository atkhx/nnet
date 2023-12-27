#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

struct Parameters {
    uint width;
};

@implementation MPSCustomKernelImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _copyPSO;
    id<MTLComputePipelineState> _copyWHDPSO;
    id<MTLComputePipelineState> _fillPSO;
    id<MTLComputePipelineState> _addPSO;
    id<MTLComputePipelineState> _addToPSO;
    id<MTLComputePipelineState> _addToWHDPSO;
    id<MTLComputePipelineState> _addToWHDBwdPSO;
    id<MTLComputePipelineState> _addScalarPSO;
    id<MTLComputePipelineState> _mulPSO;
    id<MTLComputePipelineState> _subMaxByRowPSO;
    id<MTLComputePipelineState> _divOnSumPSO;
    id<MTLComputePipelineState> _expPSO;
    id<MTLComputePipelineState> _maxByRowPSO;
    id<MTLComputePipelineState> _sumByRowPSO;
    id<MTLComputePipelineState> _nllByPosPSO;
    id<MTLComputePipelineState> _nllByPosBwdPOS;
    id<MTLComputePipelineState> _crossEntropyPosBwdPOS;
    id<MTLComputePipelineState> _updateWithAdamPSO;
    id<MTLComputePipelineState> _softmaxTrilPSO;
    id<MTLComputePipelineState> _softmaxTrilBwdPSO;
    id<MTLComputePipelineState> _transposeToPSO;
    id<MTLComputePipelineState> _transposeAndAddToPSO;

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

//     MTLSize maxThreadsPerThreadgroup = [_device maxThreadsPerThreadgroup];
//
//     // Вывод информации
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.width);
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.height);
//     NSLog(@"Максимальное количество потоков в рабочей группе: %lu", maxThreadsPerThreadgroup.depth);

    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _copyPSO = [self createPipelineStateWithFunctionName:@"copy"];
        _copyWHDPSO = [self createPipelineStateWithFunctionName:@"copyWHD"];
        _fillPSO = [self createPipelineStateWithFunctionName:@"fill"];
        _addPSO = [self createPipelineStateWithFunctionName:@"add"];
        _addToPSO = [self createPipelineStateWithFunctionName:@"addTo"];
        _addToWHDPSO = [self createPipelineStateWithFunctionName:@"addToWHD"];
        _addToWHDBwdPSO = [self createPipelineStateWithFunctionName:@"addToWHDBwd"];
        _addScalarPSO = [self createPipelineStateWithFunctionName:@"addScalar"];
        _mulPSO = [self createPipelineStateWithFunctionName:@"mul"];
        _updateWithAdamPSO = [self createPipelineStateWithFunctionName:@"updateWithAdam"];

        _expPSO = [self createPipelineStateWithFunctionName:@"exp"];
        _sumByRowPSO = [self createPipelineStateWithFunctionName:@"sumByRow"];
        _nllByPosPSO = [self createPipelineStateWithFunctionName:@"nllByPos"];
        _nllByPosBwdPOS = [self createPipelineStateWithFunctionName:@"nllByPosBwd"];
        _maxByRowPSO = [self createPipelineStateWithFunctionName:@"maxByRow"];
        _divOnSumPSO = [self createPipelineStateWithFunctionName:@"divOnSum"];
        _subMaxByRowPSO = [self createPipelineStateWithFunctionName:@"subMaxByRow"];

        _crossEntropyPosBwdPOS = [self createPipelineStateWithFunctionName:@"crossEntropyPosBwd"];

        _softmaxTrilPSO = [self createPipelineStateWithFunctionName:@"softmaxTril"];
        _softmaxTrilBwdPSO = [self createPipelineStateWithFunctionName:@"softmaxBufferTrilBwd"];

        _transposeToPSO = [self createPipelineStateWithFunctionName:@"transposeTo"];
        _transposeAndAddToPSO = [self createPipelineStateWithFunctionName:@"transposeAndAddTo"];
    }
    return self;
}

- (void) copy:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:srcBuffer
                  sourceOffset:srcOffset
                      toBuffer:dstBuffer
             destinationOffset:dstOffset
                          size:length];
    [blitEncoder endEncoding];

//     id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
//
//     [computeEncoder setComputePipelineState:_copyPSO];
//     [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
//     [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
//     [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
//     [computeEncoder endEncoding];
}

- (void) copyWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        W:(uint)W
        H:(uint)H
        D:(uint)D
{

    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:srcBuffer
                  sourceOffset:0
                      toBuffer:dstBuffer
             destinationOffset:0
                          size:srcBuffer.length];
    [blitEncoder endEncoding];

//     int square = W * H;
//
//     id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
//     [computeEncoder setComputePipelineState:_copyWHDPSO];
//     [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
//     [computeEncoder setBuffer:srcBuffer offset:0 atIndex:1];
//     [computeEncoder setBytes:&W length:sizeof(uint) atIndex:2];
//     [computeEncoder setBytes:&H length:sizeof(uint) atIndex:3];
//     [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];
//     [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
//     [computeEncoder endEncoding];
}

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_fillPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) add:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addTo:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:aBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:bBuffer offset:0 atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addToWHD:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        aBuffer:(id<MTLBuffer>)aBuffer
        bBuffer:(id<MTLBuffer>)bBuffer
        K:(float)K
        W:(uint)W
        H:(uint)H
        D:(uint)D
{
    uint square = W * H;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToWHDPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:aBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:bBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&K length:sizeof(float) atIndex:3];
    [computeEncoder setBytes:&W length:sizeof(uint) atIndex:4];
    [computeEncoder setBytes:&square length:sizeof(uint) atIndex:5];
    [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addToWHDBwd:(id<MTLCommandBuffer>)commandBuffer
        aGrad:(id<MTLBuffer>)aGrad
        bGrad:(id<MTLBuffer>)bGrad
        oGrad:(id<MTLBuffer>)oGrad
        W:(uint)W
        H:(uint)H
        D:(uint)D
{
    uint square = W * H;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addToWHDBwdPSO];
    [computeEncoder setBuffer:aGrad offset:0 atIndex:0];
    [computeEncoder setBuffer:bGrad offset:0 atIndex:1];
    [computeEncoder setBuffer:oGrad offset:0 atIndex:2];
    [computeEncoder setBytes:&W length:sizeof(uint) atIndex:3];
    [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];
    [computeEncoder dispatchThreads:MTLSizeMake(W, H, D) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) addScalar:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_addScalarPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBytes:&value length:sizeof(float) atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(dstBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) mul:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstOffset:(uint)dstOffset
        srcOffset:(uint)srcOffset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_mulPSO];
    [computeEncoder setBuffer:dstBuffer offset:dstOffset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:srcOffset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) exp:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_expPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount * rowsCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) maxByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_maxByRowPSO];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) subMaxByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        maxBuffer:(id<MTLBuffer>)maxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_subMaxByRowPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:maxBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) sumByRow:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_sumByRowPSO];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
//     [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) divOnSum:(id<MTLCommandBuffer>)commandBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_divOnSumPSO];
    [computeEncoder setBuffer:srcBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:sumBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmax:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    [self maxByRow:commandBuffer
        dstBuffer:sumBuffer
        srcBuffer:srcBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self subMaxByRow:commandBuffer
        dstBuffer:srcBuffer
        maxBuffer:sumBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self exp:commandBuffer
        dstBuffer:dstBuffer
        srcBuffer:srcBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self sumByRow:commandBuffer
        dstBuffer:sumBuffer
        srcBuffer:dstBuffer
        colsCount:colsCount
        rowsCount:rowsCount
        offset:offset];

    [self divOnSum:commandBuffer
        srcBuffer:dstBuffer
        dstBuffer:dstBuffer
        sumBuffer:sumBuffer
        colsCount:colsCount
        rowsCount:rowsCount];
}

- (void) nllByPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_nllByPosPSO];
    [computeEncoder setBuffer:dstBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:smxBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:tgtBuffer offset:0 atIndex:2];
    [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(tgtBuffer.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) nllByPosBwd:(id<MTLCommandBuffer>)commandBuffer
        oGrad:(id<MTLBuffer>)oGrad
        aGrad:(id<MTLBuffer>)aGrad
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_nllByPosBwdPOS];
        [computeEncoder setBuffer:oGrad offset:0 atIndex:0];
        [computeEncoder setBuffer:aGrad offset:0 atIndex:1];
        [computeEncoder setBuffer:tgtBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:smxBuffer offset:0 atIndex:3];
        [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:4];
        [computeEncoder dispatchThreads:MTLSizeMake(aGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) crossEntropyPos:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        sumBuffer:(id<MTLBuffer>)sumBuffer
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        chunkSize:(uint)chunkSize
{
    [self softmax:commandBuffer
        dstBuffer:smxBuffer
        srcBuffer:srcBuffer
        sumBuffer:sumBuffer
        colsCount:chunkSize
        rowsCount:srcBuffer.length / (chunkSize*4)
        offset:0
    ];

    [self nllByPos:commandBuffer
        dstBuffer:dstBuffer
        smxBuffer:smxBuffer
        tgtBuffer:tgtBuffer
        chunkSize:chunkSize
    ];
}

- (void) crossEntropyPosBwd:(id<MTLCommandBuffer>)commandBuffer
        oGrad:(id<MTLBuffer>)oGrad
        aGrad:(id<MTLBuffer>)aGrad
        tgtBuffer:(id<MTLBuffer>)tgtBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_crossEntropyPosBwdPOS];
        [computeEncoder setBuffer:oGrad offset:0 atIndex:0];
        [computeEncoder setBuffer:aGrad offset:0 atIndex:1];
        [computeEncoder setBuffer:tgtBuffer offset:0 atIndex:2];
        [computeEncoder setBuffer:smxBuffer offset:0 atIndex:3];
        [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:4];
        [computeEncoder dispatchThreads:MTLSizeMake(aGrad.length/4, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_updateWithAdamPSO];
    [computeEncoder setBuffer:dataBuffer offset:0 atIndex:0];
    [computeEncoder setBuffer:gradBuffer offset:0 atIndex:1];
    [computeEncoder setBuffer:mBuffer    offset:0 atIndex:2];
    [computeEncoder setBuffer:vBuffer    offset:0 atIndex:3];
    [computeEncoder setBytes:&beta1 length:sizeof(float) atIndex:4];
    [computeEncoder setBytes:&beta2 length:sizeof(float) atIndex:5];
    [computeEncoder setBytes:&beta1powIterationLR length:sizeof(float) atIndex:6];
    [computeEncoder setBytes:&beta2powIteration length:sizeof(float) atIndex:7];
    [computeEncoder dispatchThreads:MTLSizeMake(dataBuffer.length / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmaxTril:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_softmaxTrilPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) softmaxTrilBwd:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        srcBuffer:(id<MTLBuffer>)srcBuffer
        smxBuffer:(id<MTLBuffer>)smxBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:_softmaxTrilBwdPSO];
    [computeEncoder setBuffer:dstBuffer offset:offset atIndex:0];
    [computeEncoder setBuffer:srcBuffer offset:offset atIndex:1];
    [computeEncoder setBuffer:smxBuffer offset:offset atIndex:2];
    [computeEncoder setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [computeEncoder dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}

- (void) transposeTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height
{
    int square = width * height;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_transposeToPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];

        [computeEncoder setBytes:&width length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&height length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];

        [computeEncoder dispatchThreads:MTLSizeMake(width, height, outputData.length/(4*square)) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

- (void) transposeAndAddTo:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height
{
    int square = width * height;

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:_transposeAndAddToPSO];
        [computeEncoder setBuffer:inputData offset:0 atIndex:0];
        [computeEncoder setBuffer:outputData offset:0 atIndex:1];

        [computeEncoder setBytes:&width length:sizeof(uint) atIndex:2];
        [computeEncoder setBytes:&height length:sizeof(uint) atIndex:3];
        [computeEncoder setBytes:&square length:sizeof(uint) atIndex:4];

        [computeEncoder dispatchThreads:MTLSizeMake(width, height, outputData.length/(4*square)) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [computeEncoder endEncoding];
}

@end