// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --split-input-file -hipsr-insert-domain-stream-syncs %s | FileCheck %s

// A domain that reads another domain's result waits on the stream first.
// CHECK-LABEL: func.func @consumer_waits(
// CHECK-SAME:      %[[CTX:.*]]: !hipsr.context,
// CHECK-SAME:      %[[INPUT:.*]]: tensor<4xf32>) -> tensor<4xf32> {
// CHECK-NOT:       hipsr.stream_sync
// CHECK:           %[[PROD:.*]] = hipsr.pool_domain(%[[CTX]], %[[INPUT]] : !hipsr.context, tensor<4xf32>) {
// CHECK:           hipsr.stream_sync(%[[CTX]]) : !hipsr.context
// CHECK-NEXT:      %[[CONS:.*]] = hipsr.pool_domain(%[[CTX]], %[[PROD]] : !hipsr.context, tensor<4xf32>) {
// CHECK:           return %[[CONS]] : tensor<4xf32>
func.func @consumer_waits(%ctx: !hipsr.context, %input: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = hipsr.pool_domain(%ctx, %input : !hipsr.context, tensor<4xf32>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
    hipsr.pool_domain_yield %domain_input : tensor<4xf32>
  } -> tensor<4xf32> {domain_id = 0 : i64}
  %1 = hipsr.pool_domain(%ctx, %0 : !hipsr.context, tensor<4xf32>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
    hipsr.pool_domain_yield %domain_input : tensor<4xf32>
  } -> tensor<4xf32> {domain_id = 1 : i64}
  return %1 : tensor<4xf32>
}

// -----

// Operands that are only the context and graph inputs need no wait.
// CHECK-LABEL: func.func @inputs_only(
// CHECK-NOT: hipsr.stream_sync
func.func @inputs_only(%ctx: !hipsr.context, %input: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = hipsr.pool_domain(%ctx, %input : !hipsr.context, tensor<4xf32>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
    hipsr.pool_domain_yield %domain_input : tensor<4xf32>
  } -> tensor<4xf32> {domain_id = 0 : i64}
  return %0 : tensor<4xf32>
}

// -----

// A wait already sitting before the consumer is not duplicated.
// CHECK-LABEL: func.func @already_synced(
// CHECK:           hipsr.stream_sync(%{{.*}}) : !hipsr.context
// CHECK-NEXT:      hipsr.pool_domain
// CHECK-NOT:       hipsr.stream_sync
func.func @already_synced(%ctx: !hipsr.context, %input: tensor<4xf32>)
    -> tensor<4xf32> {
  %0 = hipsr.pool_domain(%ctx, %input : !hipsr.context, tensor<4xf32>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
    hipsr.pool_domain_yield %domain_input : tensor<4xf32>
  } -> tensor<4xf32> {domain_id = 0 : i64}
  hipsr.stream_sync(%ctx) : !hipsr.context
  %1 = hipsr.pool_domain(%ctx, %0 : !hipsr.context, tensor<4xf32>) {
  ^bb0(%domain_ctx: !hipsr.context, %domain_input: tensor<4xf32>):
    hipsr.pool_domain_yield %domain_input : tensor<4xf32>
  } -> tensor<4xf32> {domain_id = 1 : i64}
  return %1 : tensor<4xf32>
}
