// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --convert-to-llvm | FileCheck %s

// Lower the wait to the runtime stream-sync call.
// CHECK:       llvm.func @hipdnn_ep_stream_sync(!llvm.ptr) -> i32
// CHECK-LABEL: llvm.func @wait_stream(
// CHECK-SAME:    %[[CTX:[^:]+]]: !llvm.ptr) {
// CHECK-NEXT:    %{{.+}} = llvm.call @hipdnn_ep_stream_sync(%[[CTX]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return
// CHECK-NEXT:  }
func.func @wait_stream(%ctx: !hipsr.context) {
  hipsr.stream_sync(%ctx) : !hipsr.context
  return
}
