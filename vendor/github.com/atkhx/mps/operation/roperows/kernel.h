#ifndef ropeRowsKernel_h
#define ropeRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol ropeRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

@end


@interface ropeRowsKernelImpl : NSObject <ropeRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* ropeRowsKernel_h */
