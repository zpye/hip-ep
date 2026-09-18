/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- InsertDomainStreamSyncs.cpp - Wait before a consumer pool domain ---===//
//
// Inserts hipsr.stream_sync in the parent block before each pool_domain that
// reads another pool_domain's result, so host shape math waits for GPU work.
//
// Before:
//   %0 = hipsr.pool_domain(%ctx, %input) { ... }
//   %1 = hipsr.pool_domain(%ctx, %0) { ... }
// After:
//   %0 = hipsr.pool_domain(%ctx, %input) { ... }
//   hipsr.stream_sync(%ctx)
//   %1 = hipsr.pool_domain(%ctx, %0) { ... }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_INSERTDOMAINSTREAMSYNCSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct InsertDomainStreamSyncsPass
    : impl::InsertDomainStreamSyncsPassBase<InsertDomainStreamSyncsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getBody().empty()) {
      return;
    }

    IRRewriter rewriter(&getContext());
    for (Operation &op : llvm::make_early_inc_range(func.getBody().front())) {
      auto domainOp = dyn_cast<PoolDomainOp>(&op);
      if (!domainOp) {
        continue;
      }
      if (!llvm::any_of(domainOp.getOperands(), [](Value operand) {
            return isa_and_nonnull<PoolDomainOp>(operand.getDefiningOp());
          })) {
        continue;
      }
      Operation *prev = domainOp->getPrevNode();
      if (prev && isa<StreamSyncOp>(prev)) {
        continue;
      }
      rewriter.setInsertionPoint(domainOp);
      StreamSyncOp::create(rewriter, domainOp.getLoc(), domainOp.getOperand(0));
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
