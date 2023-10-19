#ifndef KernelMTLBufferFill_h
#define KernelMTLBufferFill_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol KernelMTLBufferFill <NSObject>
- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
- (void) fill:(id<MTLBuffer>)buffer withValue:(float)value commandBuffer:(id<MTLCommandBuffer>)commandBuffer;
- (void) fillPart:(id<MTLBuffer>)buffer withValue:(float)value commandBuffer:(id<MTLCommandBuffer>)commandBuffer offset:(uint)offset length:(uint)length;
@end


@interface KernelMTLBufferFillImpl : NSObject <KernelMTLBufferFill>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* KernelMTLBufferFill_h */
