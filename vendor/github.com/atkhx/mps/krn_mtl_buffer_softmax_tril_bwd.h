#ifndef KernelMTLBufferSoftmaxTrilBwd_h
#define KernelMTLBufferSoftmaxTrilBwd_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferSoftmaxTrilBwd <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) softmaxTrilBwd:(id<MTLBuffer>)destinationBuffer // iGrad
        sourceBuffer:(id<MTLBuffer>)sourceBuffer // oGrad
        softmaxBuffer:(id<MTLBuffer>)softmaxBuffer
//        softmaxGradBuffer:(id<MTLBuffer>)softmaxGradBuffer
//        sumOutBuffer:(id<MTLBuffer>)sumOutBuffer
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
        offset:(uint)offset
        withCommandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@end


@interface KernelMTLBufferSoftmaxTrilBwdImpl : NSObject <KernelMTLBufferSoftmaxTrilBwd>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferSoftmaxTrilBwd_h */
