#ifndef RmsRowsKernel_h
#define RmsRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RmsRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        aggData:(id<MTLBuffer>)aggData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        aggData:(id<MTLBuffer>)aggData
        aggGrad:(id<MTLBuffer>)aggGrad
        chunkSize:(uint)chunkSize;

@end


@interface RmsRowsKernelImpl : NSObject <RmsRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* RmsRowsKernel_h */
