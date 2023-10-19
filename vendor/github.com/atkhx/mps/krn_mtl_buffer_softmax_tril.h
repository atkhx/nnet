#ifndef KernelMTLBufferSoftmaxTril_h
#define KernelMTLBufferSoftmaxTril_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferSoftmaxTril <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) softmaxTril:(id<MTLBuffer>)destinationBuffer
        sourceBuffer:(id<MTLBuffer>)sourceBuffer
//        maxOutBuffer:(id<MTLBuffer>)maxOutBuffer
//        sumOutBuffer:(id<MTLBuffer>)sumOutBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferSoftmaxTrilImpl : NSObject <KernelMTLBufferSoftmaxTril>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferSoftmaxTril_h */
