//===- MemRefUtils.cpp - Utilities to support the MemRef dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the MemRef dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"


struct OperationWithOrder {
  Operation Op;
  int order;
};

void ReplaceOpwithAtomicRMW(mlir::AffineForOp forOp) {
  // Operate only on the innermost affine.for
  if(isa<AffineForOp>(forOp.getParentOp()))
    return;

  std::unordered_map<memref::MemRefOp, std::vector<OperationWithOrder>> memrefOperationGroups;
  memref::LoadOp lastLoadOp;
  memref::StoreOp lastStoreOp;
  memref::MemRefOp lastMemrefLoc;
  int order = 0;

  forOp.walk([&walk](Operation* Op) {
    if(auto loadOp = dyn_cast<memref::LoadOp>(Op)) {
      lastMemrefLoc = loadOp.getMemRef();
      memrefOperationGroups[lastMemrefLoc].push_back({loadOp, order});
    }

    // Check for all possible consecutive possible reducible arithmetic
    // Operation; Also the ArithOp needs to be immediately consecutive
    // To the load operation.
    else if(
      lastMemrefLoc &&
      memrefOperationGroups.find(lastMemrefLoc) != memrefOperationGroups.end() &&
      memrefOperationGroups[lastMemrefLoc].size() > 0 &&
      order == (memrefOperationGroups[lastMemrefLoc][0].order + 1)
    ){
      auto loadOp = memrefOperationGroups[lastMemrefLoc][0];
      if(auto arithOp = dyn_cast<arith::AddFOp>(Op))
        if(arithOp->getOperands()[1] == loadOp)
          memrefOperationGroups[lastMemrefLoc].push_back({arithOp, order});
      
      else if(auto arithOp = dyn_cast<arith::MulFOp>(Op))
        if(arithOp->getOperands()[1] == loadOp)
          memrefOperationGroups[lastMemrefLoc].push_back({arithOp, order});

      else if(auto arithOp = dyn_cast<arith::MaxFOp>(Op))
        if(arithOp->getOperands()[1] == loadOp)
          memrefOperationGroups[lastMemrefLoc].push_back({arithOp, order});

      else if(auto arithOp = dyn_cast<arith::MinFOp>(Op))
        if(arithOp->getOperands()[1] == loadOp)
          memrefOperationGroups[lastMemrefLoc].push_back({arithOp, order});
    }

    // Store the final store operation to be removed later.
    // The Store needs to be consecutive to the Arithmetic operation
    else if(
      lastMemrefLoc &&
      memrefOperationGroups.find(lastMemrefLoc) != memrefOperationGroups.end() &&
      memrefOperationGroups[lastMemrefLoc].size() > 1 &&
      order == (memrefOperationGroups[lastMemrefLoc][1].order + 1) &&
      auto storeOp = dyn_cast<memref::StoreOp>(Op) &&
      storeOp.getMemRef() == lastMemrefLoc
      )
      memrefOperationGroups[storeOp.getMemRef()].push_back({storeOp, order});
    
    order++;
  });

  // Add a check here to see if the size of the group is exactly 3.
  for(auto &group : memrefOperationGroups) {
    // Get the memory location accessed 
    mlir::Value *memoryLocation = group.first;

    std::vector<OperationWithOrder> memrefOperations = group.second;
    mlir::OpBuilder builder(memrefOperations[0].Op->getLoc());

    mlir::Operation *atomicRMW = 
      builder.create<memref::AtomicRMWOp>(
        memoryLocation,
        // Change this dynamically depending on the arithmetic operation
        AtomicRMWKind::addf
        memrefOperations[1].Op->getOperand(1)
      );

    // Need to remove all memref operations and just insert the one in last.
    for(mlir::OperationWithOrder memrefOperation : memrefOperations) {
      memrefOperation.Op->erase();
      memrefOperation.Op->dropAllReferences();
    }
  }
}