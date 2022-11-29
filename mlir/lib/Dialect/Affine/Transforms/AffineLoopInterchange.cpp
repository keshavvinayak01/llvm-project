//===----------------------------------------------------------------------===//
//
// This file implements an Interchanger for affine loop-like ops.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include <algorithm>
#include <unordered_map>
/* ***************************** */
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/IR/AffineExpr.h"
/* ***************************** */

namespace mlir {
#define GEN_PASS_DEF_AFFINELOOPINTERCHANGE
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
    struct LoopInterchange: public impl::AffineLoopInterchangeBase<LoopInterchange> {
    void runOnOperation() override;
    };
}
// Assumes that the looops are sorted from innermost to outermost
// There's a utility that checks for parallelizability, utilize it into your solution
// Handle the if-else edge case (Should breakout, call it unoptimizable)
// Valid interchange but not passing test case? -> Check what's wrong.
int calculateSpatialProfit(std::vector<Value> inductionVars, std::vector<std::vector<Value>> MemrefIndexVector) {
  int totalProfit = 0;

  int numInductionVars = 0;
  int orderMatch = 0;
  while(inductionVars[numInductionVars++]);
  numInductionVars--;

  for(int i = 0 ; i < MemrefIndexVector.size(); i++) {
    int j = 0;
    int index = 0;
    orderMatch = 0;
    while(j < MemrefIndexVector[i].size() && index < numInductionVars ) {
      while(inductionVars[index] != MemrefIndexVector[i][j] && index < numInductionVars) {
        index++;
      }
      while(inductionVars[index] == MemrefIndexVector[i][j] && index < numInductionVars) {
        orderMatch += 1;
        j++; index++;
      }
    }
    totalProfit += orderMatch;
  }
  return totalProfit;
}

void testAndPermuteLoops(SmallVector<AffineForOp, 10> forOpsLoops, SmallVector<unsigned, 10> permMap) {
  for(auto forOp: forOpsLoops) {
    SmallVector<AffineForOp, 10> nest;
    getPerfectlyNestedLoops(nest, forOp);
    if (nest.size() >= 2 && nest.size() == permMap.size()) {
      permuteLoops(nest, permMap);
    }
  }
}

void LoopInterchange::runOnOperation() {
  SmallVector <AffineForOp, 10> forOpsLoops;
  SmallVector <unsigned, 10> permMap;
  std::vector <std::vector<Value>> MemrefIndexVector; 
  int maxSpatialProfit = -1;
  SmallVector <unsigned, 10> dominantPermMap;
  func::FuncOp func = getOperation();

  int i = 0;
  func.walk([&](AffineForOp forOp) {
    // Add a check here to see if this forOp contains an if condition.
    forOpsLoops.push_back(forOp);
    permMap.push_back(i++);
  });

  // Check if there is any if condition in the outermost loop
  forOpsLoops[forOpsLoops.size()-1].walk([&](Operation *op) {
    if(isa<AffineIfOp>(op)) return;
  });
  //   Push the indices of all Load Operations
  forOpsLoops[forOpsLoops.size()-1].walk([&](AffineLoadOp loadOp) {
      std::vector<Value> operationVector;
      for(auto operand: loadOp.getOperands()) {
        if(operand.getType().isIndex()) {
          operationVector.push_back(operand);
        }
      }
      MemrefIndexVector.push_back(operationVector);
  });

  // Push the indices of all Store Operations
  forOpsLoops[forOpsLoops.size()-1].walk([&](AffineStoreOp storeOp) {
      std::vector<Value> operationVector;
      for(auto operand: storeOp.getOperands()) {
        if(operand.getType().isIndex()) {
          operationVector.push_back(operand);
        }
      }
       MemrefIndexVector.push_back(operationVector);
  });

  do {
    if (isValidLoopInterchangePermutation(forOpsLoops, permMap)) {
      std::vector<Value> inductionVars(10);
      for(int i = 0 ; i < permMap.size(); i++) {
        inductionVars[permMap[i]] = forOpsLoops[i].getInductionVar();
      }
      int tempProfit = calculateSpatialProfit(inductionVars, MemrefIndexVector);
      if(tempProfit > maxSpatialProfit) {
        maxSpatialProfit = tempProfit;
        dominantPermMap = permMap;
      }
    }
  } while(std::next_permutation(permMap.begin(), permMap.end()));

  // Rearrange the permutation
  std::reverse(dominantPermMap.begin(), dominantPermMap.end());
  if(isValidLoopInterchangePermutation(forOpsLoops, dominantPermMap)) {
    testAndPermuteLoops(forOpsLoops, dominantPermMap);
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineLoopInterchangePass() {
  return std::make_unique<LoopInterchange>();
}