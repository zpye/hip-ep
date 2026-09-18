/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

void StreamSyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // DCE treats a no-result op as dead unless it writes.
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

namespace {

constexpr const char *kStreamSync = "hipdnn_ep_stream_sync";

struct StreamSyncLowering : ConvertOpToLLVMPattern<StreamSyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StreamSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using SyncCall = RuntimeFunc<i32, hostPtr>;
    auto syncFunc = SyncCall::lookupOrCreateFn(
        rewriter, op.getLoc(), op->getParentOfType<ModuleOp>(), kStreamSync);
    if (failed(syncFunc)) {
      return failure();
    }
    if (failed(syncFunc->call(adaptor.getCtx()))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrStreamSyncLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<StreamSyncLowering>(converter);
}
