//===- StdExpandDivs.cpp - Code to prepare Std for lowering Divs to LLVM  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass to replace instances of Read-Modify-Store to AtomicRMWs present in all 
// Affine.for ops in the program. Optionally, can pass a specific Affine.for to 
// apply this transformation to.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <vector>

namespace mlir {
  namespace memref {
    #define GEN_PASS_DEF_CONVERTTOATOMICRMW
    #include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
  } // namespace memref
} // namespace mlir

#define PASS_NAME "convert-to-atomicrmw"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace {
    struct ConvertToAtomicRMW: 
      public memref::impl::ConvertToAtomicRMWBase<ConvertToAtomicRMW> {
    void runOnOperation() override;
    };
}

void ConvertToAtomicRMW::runOnOperation() {
  MLIRContext *context = &getContext();
  getOperation()->walk([&](AffineForOp forOp) {
    ReplaceOpwithAtomicRMW(forOp, context);
  });
}

std::unique_ptr<Pass> mlir::memref::createConvertToAtomicRMW() {
  return std::make_unique<ConvertToAtomicRMW>();
}