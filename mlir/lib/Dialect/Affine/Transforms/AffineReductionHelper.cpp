//===- SimplifyAffineStructures.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to simplify affine structures in operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_AFFINEREDUCTIONHELPER
#include "mlir/Dialect/Affine/Passes.h.inc"
}

#define PASS_NAME "affine-reduction-helper"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace {
    struct ReductionHelper: public impl::AffineReductionHelperBase<ReductionHelper> {
    void runOnOperation() override;
    };
}

void ReductionHelper::runOnOperation() {
  MLIRContext *context = &getContext();
  getOperation().walk([&](AffineForOp ForOp) {
    // Add a check here to see if this ForOp contains an if condition.
    if(isReductionNest(ForOp)) {
      ReplaceOpwithAtomicRMW(ForOp, context);
    };
  });

  // Done to change Affine Load/Stores to memref Load/Stores to aid GPU Dialect Translation
  RewritePatternSet patterns(context);
  populateAffineToMemrefConversionPatterns(patterns);
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect>();
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineReductionHelperPass() {
  return std::make_unique<ReductionHelper>();
}