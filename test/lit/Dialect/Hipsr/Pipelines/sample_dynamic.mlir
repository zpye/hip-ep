// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The same graph as sample_static.mlir, but with a dynamic leading extent. It
// enters the shape graph as a memref.dim. The checks cover the full LLVM IR
// after --hipsr-pipeline.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// CHECK-LABEL: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0
// CHECK-SAME: 64>
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 32
// CHECK-SAME: 6>
// CHECK-SAME: hipdnn.input_count = 2 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 2
// CHECK-SAME: 4>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -1
// CHECK-SAME: 3>
// CHECK-SAME: array<i64: -1
// CHECK-SAME: 4>]
// CHECK-SAME: hipdnn.num_op_state_slots = 2 : i32
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 4>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: -1
// CHECK-SAME: 2>]
// CHECK: llvm.mlir.global internal constant @__metadata_json
// CHECK: llvm.mlir.global internal constant @__metadata_blob
// CHECK:       llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @hipdnn_ep_stream_sync(!llvm.ptr) -> i32
// CHECK-NEXT:  llvm.func @wrap_cast(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_hipblasLtMatmul(!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @free(!llvm.ptr)
// CHECK-NEXT:  llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @hipdnn_ep_op_state_construct_matmul(!llvm.ptr, i32) -> i8
// CHECK-NEXT:  llvm.func @hipdnn_ep_op_states_alloc(!llvm.ptr, i64) -> i8
// CHECK-LABEL: llvm.func private @main_graph(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr) -> i32 attributes {passthrough = ["noinline"]} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1:.+]] = llvm.getelementptr %[[ARG1]]{{\[}}%[[V0]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V2:.+]] = llvm.load %[[V1]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.+]] = llvm.extractvalue %[[V3]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.+]] = llvm.extractvalue %[[V3]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.+]] = llvm.extractvalue %[[V3]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.+]] = llvm.extractvalue %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.+]] = llvm.extractvalue %[[V3]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.+]] = llvm.extractvalue %[[V3]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.+]] = llvm.extractvalue %[[V3]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V12:.+]] = llvm.getelementptr %[[ARG1]]{{\[}}%[[V11]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V13:.+]] = llvm.load %[[V12]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[V14:.+]] = llvm.load %[[V13]] : !llvm.ptr -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.+]] = llvm.extractvalue %[[V14]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.+]] = llvm.extractvalue %[[V14]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V17:.+]] = llvm.extractvalue %[[V14]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V18:.+]] = llvm.extractvalue %[[V14]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.+]] = llvm.extractvalue %[[V14]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.+]] = llvm.extractvalue %[[V14]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.+]] = llvm.extractvalue %[[V14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V22:.+]] = llvm.call @main_graph_internal(%[[ARG0]], %[[V4]], %[[V5]], %[[V6]], %[[V7]], %[[V8]], %[[V9]], %[[V10]], %[[V15]], %[[V16]], %[[V17]], %[[V18]], %[[V19]], %[[V20]], %[[V21]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[V23]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func private @main_graph_internal(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr<1>, %[[ARG2:[^,]*]]: !llvm.ptr<1>, %[[ARG3:[^,]*]]: i64, %[[ARG4:[^,]*]]: i64, %[[ARG5:[^,]*]]: i64, %[[ARG6:[^,]*]]: i64, %[[ARG7:[^,]*]]: i64, %[[ARG8:[^,]*]]: !llvm.ptr<1>, %[[ARG9:[^,]*]]: !llvm.ptr<1>, %[[ARG10:[^,]*]]: i64, %[[ARG11:[^,]*]]: i64, %[[ARG12:[^,]*]]: i64, %[[ARG13:[^,]*]]: i64, %[[ARG14:[^,]*]]: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> {onnx.name = "y"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1:.+]] = llvm.insertvalue %[[ARG8]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.+]] = llvm.insertvalue %[[ARG9]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.+]] = llvm.insertvalue %[[ARG10]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.+]] = llvm.insertvalue %[[ARG11]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.+]] = llvm.insertvalue %[[ARG13]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.+]] = llvm.insertvalue %[[ARG12]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.+]] = llvm.insertvalue %[[ARG14]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.+]] = llvm.insertvalue %[[ARG1]], %[[V8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.+]] = llvm.insertvalue %[[ARG2]], %[[V9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.+]] = llvm.insertvalue %[[ARG3]], %[[V10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.+]] = llvm.insertvalue %[[ARG4]], %[[V11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.+]] = llvm.insertvalue %[[ARG6]], %[[V12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.+]] = llvm.insertvalue %[[ARG5]], %[[V13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.+]] = llvm.insertvalue %[[ARG7]], %[[V14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.+]] = llvm.mlir.constant(16 : index) : i64
// CHECK-NEXT:    %[[V17:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V18:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V19:.+]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT:    %[[V20:.+]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V21:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V22:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V21]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V23:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V24:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V25:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V26:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V27:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V28:.+]] = llvm.insertvalue %[[V22]], %[[V27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V29:.+]] = llvm.insertvalue %[[V22]], %[[V28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V30:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V31:.+]] = llvm.insertvalue %[[V30]], %[[V29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V32:.+]] = llvm.insertvalue %[[V23]], %[[V31]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V33:.+]] = llvm.insertvalue %[[V24]], %[[V32]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V34:.+]] = llvm.insertvalue %[[V26]], %[[V33]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V35:.+]] = llvm.insertvalue %[[V25]], %[[V34]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V36:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V37:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V36]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V38:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V39:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V40:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V41:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V42:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V43:.+]] = llvm.insertvalue %[[V37]], %[[V42]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V44:.+]] = llvm.insertvalue %[[V37]], %[[V43]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V45:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V46:.+]] = llvm.insertvalue %[[V45]], %[[V44]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V47:.+]] = llvm.insertvalue %[[V38]], %[[V46]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V48:.+]] = llvm.insertvalue %[[V39]], %[[V47]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V49:.+]] = llvm.insertvalue %[[V41]], %[[V48]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V50:.+]] = llvm.insertvalue %[[V40]], %[[V49]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V51:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V52:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V53:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V54:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V55:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V56:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V57:.+]] = llvm.getelementptr %[[V56]]{{\[}}%[[V54]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V58:.+]] = llvm.ptrtoint %[[V57]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V59:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V60:.+]] = llvm.add %[[V58]], %[[V59]] : i64
// CHECK-NEXT:    %[[V61:.+]] = llvm.call @malloc(%[[V60]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V62:.+]] = llvm.ptrtoint %[[V61]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V63:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V64:.+]] = llvm.sub %[[V59]], %[[V63]] : i64
// CHECK-NEXT:    %[[V65:.+]] = llvm.add %[[V62]], %[[V64]] : i64
// CHECK-NEXT:    %[[V66:.+]] = llvm.urem %[[V65]], %[[V59]] : i64
// CHECK-NEXT:    %[[V67:.+]] = llvm.sub %[[V65]], %[[V66]] : i64
// CHECK-NEXT:    %[[V68:.+]] = llvm.inttoptr %[[V67]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V69:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V70:.+]] = llvm.insertvalue %[[V61]], %[[V69]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V71:.+]] = llvm.insertvalue %[[V68]], %[[V70]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V72:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V73:.+]] = llvm.insertvalue %[[V72]], %[[V71]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V74:.+]] = llvm.insertvalue %[[V54]], %[[V73]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V75:.+]] = llvm.insertvalue %[[V55]], %[[V74]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V76:.+]] = llvm.extractvalue %[[V75]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V77:.+]] = llvm.getelementptr inbounds|nuw %[[V76]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V53]], %[[V77]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V78:.+]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V79:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V80:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V81:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V82:.+]] = llvm.getelementptr %[[V81]]{{\[}}%[[V79]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V83:.+]] = llvm.ptrtoint %[[V82]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V84:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V85:.+]] = llvm.add %[[V83]], %[[V84]] : i64
// CHECK-NEXT:    %[[V86:.+]] = llvm.call @malloc(%[[V85]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V87:.+]] = llvm.ptrtoint %[[V86]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V88:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V89:.+]] = llvm.sub %[[V84]], %[[V88]] : i64
// CHECK-NEXT:    %[[V90:.+]] = llvm.add %[[V87]], %[[V89]] : i64
// CHECK-NEXT:    %[[V91:.+]] = llvm.urem %[[V90]], %[[V84]] : i64
// CHECK-NEXT:    %[[V92:.+]] = llvm.sub %[[V90]], %[[V91]] : i64
// CHECK-NEXT:    %[[V93:.+]] = llvm.inttoptr %[[V92]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V94:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V95:.+]] = llvm.insertvalue %[[V86]], %[[V94]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V96:.+]] = llvm.insertvalue %[[V93]], %[[V95]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V97:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V98:.+]] = llvm.insertvalue %[[V97]], %[[V96]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V99:.+]] = llvm.insertvalue %[[V79]], %[[V98]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V100:.+]] = llvm.insertvalue %[[V80]], %[[V99]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V101:.+]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V102:.+]] = llvm.getelementptr inbounds|nuw %[[V101]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V78]], %[[V102]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V103:.+]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V104:.+]] = llvm.getelementptr inbounds|nuw %[[V103]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V51]], %[[V104]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V105:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V106:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V107:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V108:.+]] = llvm.getelementptr %[[V107]]{{\[}}%[[V105]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V109:.+]] = llvm.ptrtoint %[[V108]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V110:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V111:.+]] = llvm.add %[[V109]], %[[V110]] : i64
// CHECK-NEXT:    %[[V112:.+]] = llvm.call @malloc(%[[V111]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V113:.+]] = llvm.ptrtoint %[[V112]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V114:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V115:.+]] = llvm.sub %[[V110]], %[[V114]] : i64
// CHECK-NEXT:    %[[V116:.+]] = llvm.add %[[V113]], %[[V115]] : i64
// CHECK-NEXT:    %[[V117:.+]] = llvm.urem %[[V116]], %[[V110]] : i64
// CHECK-NEXT:    %[[V118:.+]] = llvm.sub %[[V116]], %[[V117]] : i64
// CHECK-NEXT:    %[[V119:.+]] = llvm.inttoptr %[[V118]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V120:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V121:.+]] = llvm.insertvalue %[[V112]], %[[V120]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V122:.+]] = llvm.insertvalue %[[V119]], %[[V121]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V123:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V124:.+]] = llvm.insertvalue %[[V123]], %[[V122]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V125:.+]] = llvm.insertvalue %[[V105]], %[[V124]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.+]] = llvm.insertvalue %[[V106]], %[[V125]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V127:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V128:.+]] = llvm.extractvalue %[[V100]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V129:.+]] = llvm.mul %[[V127]], %[[V128]] : i64
// CHECK-NEXT:    %[[V130:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V131:.+]] = llvm.getelementptr %[[V130]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V132:.+]] = llvm.ptrtoint %[[V131]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V133:.+]] = llvm.mul %[[V129]], %[[V132]] : i64
// CHECK-NEXT:    %[[V134:.+]] = llvm.extractvalue %[[V100]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V135:.+]] = llvm.extractvalue %[[V100]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V136:.+]] = llvm.getelementptr %[[V134]]{{\[}}%[[V135]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V137:.+]] = llvm.extractvalue %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V138:.+]] = llvm.extractvalue %[[V126]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V139:.+]] = llvm.getelementptr %[[V137]]{{\[}}%[[V138]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V139]], %[[V136]], %[[V133]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V140:.+]] = llvm.extractvalue %[[V100]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V140]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V141:.+]] = llvm.extractvalue %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V142:.+]] = llvm.getelementptr inbounds|nuw %[[V141]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V143:.+]] = llvm.load %[[V142]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V144:.+]] = llvm.mul %[[V143]], %[[V53]] : i64
// CHECK-NEXT:    %[[V145:.+]] = llvm.add %[[V144]], %[[V19]] : i64
// CHECK-NEXT:    %[[V146:.+]] = llvm.udiv %[[V145]], %[[V20]] : i64
// CHECK-NEXT:    %[[V147:.+]] = llvm.mul %[[V146]], %[[V20]] : i64
// CHECK-NEXT:    %[[V148:.+]] = llvm.mul %[[V143]], %[[V18]] : i64
// CHECK-NEXT:    %[[V149:.+]] = llvm.add %[[V148]], %[[V19]] : i64
// CHECK-NEXT:    %[[V150:.+]] = llvm.udiv %[[V149]], %[[V20]] : i64
// CHECK-NEXT:    %[[V151:.+]] = llvm.mul %[[V150]], %[[V20]] : i64
// CHECK-NEXT:    %[[V152:.+]] = llvm.add %[[V147]], %[[V151]] : i64
// CHECK-NEXT:    %[[V153:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V154:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V153]], %[[V152]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V155:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V156:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V157:.+]] = llvm.insertvalue %[[V154]], %[[V156]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V158:.+]] = llvm.insertvalue %[[V154]], %[[V157]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V160:.+]] = llvm.insertvalue %[[V159]], %[[V158]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.+]] = llvm.insertvalue %[[V152]], %[[V160]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V162:.+]] = llvm.insertvalue %[[V155]], %[[V161]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V163:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V164:.+]] = llvm.extractvalue %[[V162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V165:.+]] = llvm.insertvalue %[[V164]], %[[V163]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V166:.+]] = llvm.extractvalue %[[V162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V167:.+]] = llvm.getelementptr %[[V166]]{{\[}}%[[V52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V168:.+]] = llvm.insertvalue %[[V167]], %[[V165]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V169:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V170:.+]] = llvm.insertvalue %[[V169]], %[[V168]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V171:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.+]] = llvm.insertvalue %[[V171]], %[[V170]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V173:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V174:.+]] = llvm.insertvalue %[[V173]], %[[V172]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V175:.+]] = llvm.insertvalue %[[V143]], %[[V174]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V176:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V177:.+]] = llvm.insertvalue %[[V176]], %[[V175]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V178:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V179:.+]] = llvm.extractvalue %[[V162]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V180:.+]] = llvm.insertvalue %[[V179]], %[[V178]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V181:.+]] = llvm.extractvalue %[[V162]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V182:.+]] = llvm.getelementptr %[[V181]]{{\[}}%[[V147]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V183:.+]] = llvm.insertvalue %[[V182]], %[[V180]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V184:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V185:.+]] = llvm.insertvalue %[[V184]], %[[V183]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V186:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V187:.+]] = llvm.insertvalue %[[V186]], %[[V185]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V188:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V189:.+]] = llvm.insertvalue %[[V188]], %[[V187]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V190:.+]] = llvm.insertvalue %[[V143]], %[[V189]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V191:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V192:.+]] = llvm.insertvalue %[[V191]], %[[V190]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V193:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V194:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V195:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V196:.+]] = llvm.getelementptr %[[V195]]{{\[}}%[[V193]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V197:.+]] = llvm.ptrtoint %[[V196]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V198:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V199:.+]] = llvm.add %[[V197]], %[[V198]] : i64
// CHECK-NEXT:    %[[V200:.+]] = llvm.call @malloc(%[[V199]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V201:.+]] = llvm.ptrtoint %[[V200]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V202:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V203:.+]] = llvm.sub %[[V198]], %[[V202]] : i64
// CHECK-NEXT:    %[[V204:.+]] = llvm.add %[[V201]], %[[V203]] : i64
// CHECK-NEXT:    %[[V205:.+]] = llvm.urem %[[V204]], %[[V198]] : i64
// CHECK-NEXT:    %[[V206:.+]] = llvm.sub %[[V204]], %[[V205]] : i64
// CHECK-NEXT:    %[[V207:.+]] = llvm.inttoptr %[[V206]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V208:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V209:.+]] = llvm.insertvalue %[[V200]], %[[V208]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V210:.+]] = llvm.insertvalue %[[V207]], %[[V209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V211:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V212:.+]] = llvm.insertvalue %[[V211]], %[[V210]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V213:.+]] = llvm.insertvalue %[[V193]], %[[V212]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V214:.+]] = llvm.insertvalue %[[V194]], %[[V213]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V215:.+]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V216:.+]] = llvm.extractvalue %[[V15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V217:.+]] = llvm.extractvalue %[[V50]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V218:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V219:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V220:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V221:.+]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V222:.+]] = llvm.extractvalue %[[V50]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V223:.+]] = llvm.extractvalue %[[V177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V224:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V225:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V226:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V227:.+]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V220]], %[[V221]], %[[V222]], %[[V223]], %[[V215]], %[[V217]], %[[V216]], %[[V218]], %[[V224]], %[[V219]], %[[V225]], %[[V226]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V228:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V229:.+]] = llvm.extractvalue %[[V192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V230:.+]] = llvm.mul %[[V228]], %[[V229]] : i64
// CHECK-NEXT:    %[[V231:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V232:.+]] = llvm.mul %[[V230]], %[[V231]] : i64
// CHECK-NEXT:    %[[V233:.+]] = llvm.extractvalue %[[V177]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V234:.+]] = llvm.extractvalue %[[V192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V235:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V236:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V237:.+]] = llvm.call @wrap_cast(%[[ARG0]], %[[V233]], %[[V234]], %[[V232]], %[[V235]], %[[V236]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V238:.+]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V239:.+]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V240:.+]] = llvm.getelementptr inbounds|nuw %[[V239]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V238]], %[[V240]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V241:.+]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V242:.+]] = llvm.getelementptr inbounds|nuw %[[V241]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V17]], %[[V242]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V243:.+]] = llvm.extractvalue %[[V126]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V243]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V244:.+]] = llvm.extractvalue %[[V75]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V244]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V245:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V246:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V247:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V248:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V249:.+]] = llvm.getelementptr %[[V248]]{{\[}}%[[V246]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V250:.+]] = llvm.ptrtoint %[[V249]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V251:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V252:.+]] = llvm.add %[[V250]], %[[V251]] : i64
// CHECK-NEXT:    %[[V253:.+]] = llvm.call @malloc(%[[V252]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V254:.+]] = llvm.ptrtoint %[[V253]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V255:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V256:.+]] = llvm.sub %[[V251]], %[[V255]] : i64
// CHECK-NEXT:    %[[V257:.+]] = llvm.add %[[V254]], %[[V256]] : i64
// CHECK-NEXT:    %[[V258:.+]] = llvm.urem %[[V257]], %[[V251]] : i64
// CHECK-NEXT:    %[[V259:.+]] = llvm.sub %[[V257]], %[[V258]] : i64
// CHECK-NEXT:    %[[V260:.+]] = llvm.inttoptr %[[V259]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V261:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V262:.+]] = llvm.insertvalue %[[V253]], %[[V261]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V263:.+]] = llvm.insertvalue %[[V260]], %[[V262]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V264:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V265:.+]] = llvm.insertvalue %[[V264]], %[[V263]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V266:.+]] = llvm.insertvalue %[[V246]], %[[V265]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V267:.+]] = llvm.insertvalue %[[V247]], %[[V266]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V268:.+]] = llvm.extractvalue %[[V267]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V269:.+]] = llvm.getelementptr inbounds|nuw %[[V268]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V143]], %[[V269]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V270:.+]] = llvm.extractvalue %[[V267]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V271:.+]] = llvm.getelementptr inbounds|nuw %[[V270]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V51]], %[[V271]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V272:.+]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V273:.+]] = llvm.getelementptr inbounds|nuw %[[V272]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V274:.+]] = llvm.load %[[V273]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V275:.+]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V276:.+]] = llvm.getelementptr inbounds|nuw %[[V275]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V277:.+]] = llvm.load %[[V276]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V278:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V279:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V280:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V281:.+]] = llvm.getelementptr %[[V280]]{{\[}}%[[V278]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V282:.+]] = llvm.ptrtoint %[[V281]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V283:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V284:.+]] = llvm.add %[[V282]], %[[V283]] : i64
// CHECK-NEXT:    %[[V285:.+]] = llvm.call @malloc(%[[V284]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V286:.+]] = llvm.ptrtoint %[[V285]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V287:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V288:.+]] = llvm.sub %[[V283]], %[[V287]] : i64
// CHECK-NEXT:    %[[V289:.+]] = llvm.add %[[V286]], %[[V288]] : i64
// CHECK-NEXT:    %[[V290:.+]] = llvm.urem %[[V289]], %[[V283]] : i64
// CHECK-NEXT:    %[[V291:.+]] = llvm.sub %[[V289]], %[[V290]] : i64
// CHECK-NEXT:    %[[V292:.+]] = llvm.inttoptr %[[V291]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V293:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V294:.+]] = llvm.insertvalue %[[V285]], %[[V293]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V295:.+]] = llvm.insertvalue %[[V292]], %[[V294]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V296:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V297:.+]] = llvm.insertvalue %[[V296]], %[[V295]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V298:.+]] = llvm.insertvalue %[[V278]], %[[V297]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V299:.+]] = llvm.insertvalue %[[V279]], %[[V298]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V300:.+]] = llvm.extractvalue %[[V299]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V301:.+]] = llvm.getelementptr inbounds|nuw %[[V300]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V274]], %[[V301]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V302:.+]] = llvm.extractvalue %[[V299]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V303:.+]] = llvm.getelementptr inbounds|nuw %[[V302]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V277]], %[[V303]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V304:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V305:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V306:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V307:.+]] = llvm.getelementptr %[[V306]]{{\[}}%[[V304]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V308:.+]] = llvm.ptrtoint %[[V307]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V309:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V310:.+]] = llvm.add %[[V308]], %[[V309]] : i64
// CHECK-NEXT:    %[[V311:.+]] = llvm.call @malloc(%[[V310]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V312:.+]] = llvm.ptrtoint %[[V311]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V313:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V314:.+]] = llvm.sub %[[V309]], %[[V313]] : i64
// CHECK-NEXT:    %[[V315:.+]] = llvm.add %[[V312]], %[[V314]] : i64
// CHECK-NEXT:    %[[V316:.+]] = llvm.urem %[[V315]], %[[V309]] : i64
// CHECK-NEXT:    %[[V317:.+]] = llvm.sub %[[V315]], %[[V316]] : i64
// CHECK-NEXT:    %[[V318:.+]] = llvm.inttoptr %[[V317]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V319:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V320:.+]] = llvm.insertvalue %[[V311]], %[[V319]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V321:.+]] = llvm.insertvalue %[[V318]], %[[V320]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V322:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V323:.+]] = llvm.insertvalue %[[V322]], %[[V321]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V324:.+]] = llvm.insertvalue %[[V304]], %[[V323]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V325:.+]] = llvm.insertvalue %[[V305]], %[[V324]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb1(%[[V52]] : i64)
// CHECK-NEXT:    ^bb1(%[[V326:.+]]: i64):
// CHECK-NEXT:    %[[V327:.+]] = llvm.icmp "slt" %[[V326]], %[[V53]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V327]], ^bb2, ^bb7
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    %[[V328:.+]] = llvm.icmp "ult" %[[V326]], %[[V52]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V328]], ^bb3, ^bb4
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:    llvm.br ^bb5(%[[V51]] : i64)
// CHECK-NEXT:    ^bb4:
// CHECK-NEXT:    %[[V329:.+]] = llvm.extractvalue %[[V267]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V330:.+]] = llvm.getelementptr inbounds|nuw %[[V329]]{{\[}}%[[V326]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V331:.+]] = llvm.load %[[V330]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V332:.+]] = llvm.extractvalue %[[V299]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V333:.+]] = llvm.getelementptr inbounds|nuw %[[V332]]{{\[}}%[[V326]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V334:.+]] = llvm.load %[[V333]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V335:.+]] = llvm.icmp "eq" %[[V334]], %[[V51]] : i64
// CHECK-NEXT:    %[[V336:.+]] = llvm.select %[[V335]], %[[V331]], %[[V334]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb5(%[[V336]] : i64)
// CHECK-NEXT:    ^bb5(%[[V337:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:    ^bb6:
// CHECK-NEXT:    %[[V338:.+]] = llvm.extractvalue %[[V325]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V339:.+]] = llvm.getelementptr inbounds|nuw %[[V338]]{{\[}}%[[V326]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V337]], %[[V339]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V340:.+]] = llvm.add %[[V326]], %[[V51]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V340]] : i64)
// CHECK-NEXT:    ^bb7:
// CHECK-NEXT:    %[[V341:.+]] = llvm.extractvalue %[[V299]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V341]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V342:.+]] = llvm.extractvalue %[[V267]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V342]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V343:.+]] = llvm.extractvalue %[[V325]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V344:.+]] = llvm.getelementptr inbounds|nuw %[[V343]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V345:.+]] = llvm.load %[[V344]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V346:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V347:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V348:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V349:.+]] = llvm.getelementptr %[[V348]]{{\[}}%[[V346]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V350:.+]] = llvm.ptrtoint %[[V349]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V351:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V352:.+]] = llvm.add %[[V350]], %[[V351]] : i64
// CHECK-NEXT:    %[[V353:.+]] = llvm.call @malloc(%[[V352]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V354:.+]] = llvm.ptrtoint %[[V353]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V355:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V356:.+]] = llvm.sub %[[V351]], %[[V355]] : i64
// CHECK-NEXT:    %[[V357:.+]] = llvm.add %[[V354]], %[[V356]] : i64
// CHECK-NEXT:    %[[V358:.+]] = llvm.urem %[[V357]], %[[V351]] : i64
// CHECK-NEXT:    %[[V359:.+]] = llvm.sub %[[V357]], %[[V358]] : i64
// CHECK-NEXT:    %[[V360:.+]] = llvm.inttoptr %[[V359]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V361:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V362:.+]] = llvm.insertvalue %[[V353]], %[[V361]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V363:.+]] = llvm.insertvalue %[[V360]], %[[V362]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V364:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V365:.+]] = llvm.insertvalue %[[V364]], %[[V363]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V366:.+]] = llvm.insertvalue %[[V346]], %[[V365]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V367:.+]] = llvm.insertvalue %[[V347]], %[[V366]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V368:.+]] = llvm.extractvalue %[[V367]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V369:.+]] = llvm.getelementptr inbounds|nuw %[[V368]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V345]], %[[V369]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V370:.+]] = llvm.extractvalue %[[V367]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V371:.+]] = llvm.getelementptr inbounds|nuw %[[V370]]{{\[}}%[[V51]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V53]], %[[V371]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V372:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V373:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V374:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V375:.+]] = llvm.getelementptr %[[V374]]{{\[}}%[[V372]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V376:.+]] = llvm.ptrtoint %[[V375]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V377:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V378:.+]] = llvm.add %[[V376]], %[[V377]] : i64
// CHECK-NEXT:    %[[V379:.+]] = llvm.call @malloc(%[[V378]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V380:.+]] = llvm.ptrtoint %[[V379]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V381:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V382:.+]] = llvm.sub %[[V377]], %[[V381]] : i64
// CHECK-NEXT:    %[[V383:.+]] = llvm.add %[[V380]], %[[V382]] : i64
// CHECK-NEXT:    %[[V384:.+]] = llvm.urem %[[V383]], %[[V377]] : i64
// CHECK-NEXT:    %[[V385:.+]] = llvm.sub %[[V383]], %[[V384]] : i64
// CHECK-NEXT:    %[[V386:.+]] = llvm.inttoptr %[[V385]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V387:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V388:.+]] = llvm.insertvalue %[[V379]], %[[V387]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V389:.+]] = llvm.insertvalue %[[V386]], %[[V388]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V390:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V391:.+]] = llvm.insertvalue %[[V390]], %[[V389]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.+]] = llvm.insertvalue %[[V372]], %[[V391]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V393:.+]] = llvm.insertvalue %[[V373]], %[[V392]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V394:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V395:.+]] = llvm.extractvalue %[[V367]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V396:.+]] = llvm.mul %[[V394]], %[[V395]] : i64
// CHECK-NEXT:    %[[V397:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V398:.+]] = llvm.getelementptr %[[V397]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V399:.+]] = llvm.ptrtoint %[[V398]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V400:.+]] = llvm.mul %[[V396]], %[[V399]] : i64
// CHECK-NEXT:    %[[V401:.+]] = llvm.extractvalue %[[V367]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V402:.+]] = llvm.extractvalue %[[V367]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V403:.+]] = llvm.getelementptr %[[V401]]{{\[}}%[[V402]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V404:.+]] = llvm.extractvalue %[[V393]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V405:.+]] = llvm.extractvalue %[[V393]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V406:.+]] = llvm.getelementptr %[[V404]]{{\[}}%[[V405]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V406]], %[[V403]], %[[V400]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V407:.+]] = llvm.extractvalue %[[V367]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V407]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V408:.+]] = llvm.extractvalue %[[V325]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V409:.+]] = llvm.getelementptr inbounds|nuw %[[V408]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V410:.+]] = llvm.load %[[V409]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V411:.+]] = llvm.mul %[[V410]], %[[V16]] : i64
// CHECK-NEXT:    %[[V412:.+]] = llvm.add %[[V411]], %[[V19]] : i64
// CHECK-NEXT:    %[[V413:.+]] = llvm.udiv %[[V412]], %[[V20]] : i64
// CHECK-NEXT:    %[[V414:.+]] = llvm.mul %[[V413]], %[[V20]] : i64
// CHECK-NEXT:    %[[V415:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V416:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V415]], %[[V414]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V417:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V418:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V419:.+]] = llvm.insertvalue %[[V416]], %[[V418]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V420:.+]] = llvm.insertvalue %[[V416]], %[[V419]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V421:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V422:.+]] = llvm.insertvalue %[[V421]], %[[V420]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V423:.+]] = llvm.insertvalue %[[V414]], %[[V422]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V424:.+]] = llvm.insertvalue %[[V417]], %[[V423]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V425:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V426:.+]] = llvm.extractvalue %[[V424]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V427:.+]] = llvm.insertvalue %[[V426]], %[[V425]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V428:.+]] = llvm.extractvalue %[[V424]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V429:.+]] = llvm.getelementptr %[[V428]]{{\[}}%[[V52]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V430:.+]] = llvm.insertvalue %[[V429]], %[[V427]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V431:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V432:.+]] = llvm.insertvalue %[[V431]], %[[V430]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V433:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V434:.+]] = llvm.insertvalue %[[V433]], %[[V432]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V435:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V436:.+]] = llvm.insertvalue %[[V435]], %[[V434]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V437:.+]] = llvm.insertvalue %[[V410]], %[[V436]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V438:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V439:.+]] = llvm.insertvalue %[[V438]], %[[V437]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V440:.+]] = llvm.extractvalue %[[V393]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V441:.+]] = llvm.getelementptr inbounds|nuw %[[V440]]{{\[}}%[[V52]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V442:.+]] = llvm.load %[[V441]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V443:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V444:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V445:.+]] = llvm.mul %[[V443]], %[[V442]] : i64
// CHECK-NEXT:    %[[V446:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V447:.+]] = llvm.getelementptr %[[V446]]{{\[}}%[[V445]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %[[V448:.+]] = llvm.ptrtoint %[[V447]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V449:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V450:.+]] = llvm.alloca %[[V449]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V451:.+]] = llvm.getelementptr %[[V450]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V451]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V452:.+]] = llvm.getelementptr %[[V450]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V443]], %[[V452]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V453:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V454:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V455:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V456:.+]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V453]], %[[V450]], %[[V454]], %[[V455]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V457:.+]] = llvm.addrspacecast %[[V456]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V458:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V459:.+]] = llvm.insertvalue %[[V457]], %[[V458]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V460:.+]] = llvm.insertvalue %[[V457]], %[[V459]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V461:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V462:.+]] = llvm.insertvalue %[[V461]], %[[V460]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V463:.+]] = llvm.insertvalue %[[V442]], %[[V462]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V464:.+]] = llvm.insertvalue %[[V443]], %[[V463]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V465:.+]] = llvm.insertvalue %[[V443]], %[[V464]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V466:.+]] = llvm.insertvalue %[[V444]], %[[V465]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V467:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V468:.+]] = llvm.extractvalue %[[V214]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V469:.+]] = llvm.alloca %[[V467]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V470:.+]] = llvm.extractvalue %[[V192]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V471:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V472:.+]] = llvm.getelementptr %[[V469]]{{\[}}%[[V471]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V470]], %[[V472]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V473:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V474:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V475:.+]] = llvm.getelementptr %[[V469]]{{\[}}%[[V474]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V473]], %[[V475]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V476:.+]] = llvm.alloca %[[V467]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V477:.+]] = llvm.extractvalue %[[V439]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V478:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V479:.+]] = llvm.getelementptr %[[V476]]{{\[}}%[[V478]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V477]], %[[V479]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V480:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V481:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V482:.+]] = llvm.getelementptr %[[V476]]{{\[}}%[[V481]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V480]], %[[V482]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V483:.+]] = llvm.extractvalue %[[V192]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V484:.+]] = llvm.extractvalue %[[V439]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V485:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V486:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V487:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V488:.+]] = llvm.call @wrap_expand(%[[ARG0]], %[[V483]], %[[V468]], %[[V484]], %[[V469]], %[[V485]], %[[V476]], %[[V486]], %[[V487]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V489:.+]] = llvm.extractvalue %[[V214]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V489]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V490:.+]] = llvm.extractvalue %[[V439]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V491:.+]] = llvm.extractvalue %[[V439]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V492:.+]] = llvm.extractvalue %[[V35]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V493:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V494:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V495:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V496:.+]] = llvm.extractvalue %[[V439]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V497:.+]] = llvm.extractvalue %[[V35]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V498:.+]] = llvm.extractvalue %[[V466]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V499:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V500:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V501:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V502:.+]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V495]], %[[V496]], %[[V497]], %[[V498]], %[[V490]], %[[V492]], %[[V491]], %[[V493]], %[[V499]], %[[V494]], %[[V500]], %[[V501]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V503:.+]] = llvm.extractvalue %[[V325]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V503]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V504:.+]] = llvm.extractvalue %[[V393]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V504]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V466]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @hipdnn_ep_op_states_init_fn(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr) -> i32 {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V2:.+]] = llvm.mlir.constant(0 : i8) : i8
// CHECK-NEXT:    %[[V3:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V4:.+]] = llvm.call @hipdnn_ep_op_states_alloc(%[[ARG0]], %[[V3]]) : (!llvm.ptr, i64) -> i8
// CHECK-NEXT:    %[[V5:.+]] = llvm.icmp "eq" %[[V4]], %[[V2]] : i8
// CHECK-NEXT:    llvm.cond_br %[[V5]], ^bb2, ^bb1
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:    %[[V6:.+]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%[[ARG0]], %[[V1]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:    %[[V7:.+]] = llvm.call @hipdnn_ep_op_state_construct_matmul(%[[ARG0]], %[[V0]]) : (!llvm.ptr, i32) -> i8
// CHECK-NEXT:    llvm.return %[[V1]] : i32
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    llvm.return %[[V0]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @inference_init(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr, %[[ARG2:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT:    %[[V1:.+]] = llvm.mlir.constant(304 : i64) : i64
// CHECK-NEXT:    %[[V2:.+]] = llvm.call @hipdnn_ep_state_init_with_fs(%[[ARG0]], %[[ARG1]], %[[V0]], %[[V1]], %[[ARG2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[V3:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V4:.+]] = llvm.icmp "ne" %[[V2]], %[[V3]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V4]], ^bb2, ^bb1
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:    %[[V5:.+]] = llvm.load %[[ARG0]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    %[[V6:.+]] = llvm.call @hipdnn_ep_op_states_init_fn(%[[V5]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return %[[V6]] : i32
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    llvm.return %[[V2]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @inference_compute(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    llvm.call @hipdnn_ep_runtime_begin_compute(%[[ARG0]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V2:.+]] = llvm.alloca %[[V1]] x i32 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V3:.+]] = llvm.mlir.constant(48 : i64) : i64
// CHECK-NEXT:    %[[V4:.+]] = llvm.alloca %[[V3]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V5:.+]] = llvm.alloca %[[V3]] x i8 : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V6:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V7:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V8:.+]] = llvm.call @hipdnn_ep_tensor_prepare_input(%[[ARG0]], %[[ARG1]], %[[V6]], %[[V7]], %[[V4]]) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[V9:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V10:.+]] = llvm.icmp "ne" %[[V8]], %[[V9]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V10]], ^bb3, ^bb2
// CHECK-NEXT:    ^bb1:
// CHECK-NEXT:    %[[V11:.+]] = llvm.load %[[V2]] : !llvm.ptr -> i32
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[V4]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[V5]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V11]] : i32
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    %[[V12:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V13:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V14:.+]] = llvm.call @hipdnn_ep_tensor_prepare_input(%[[ARG0]], %[[ARG1]], %[[V12]], %[[V13]], %[[V5]]) : (!llvm.ptr, !llvm.ptr, i64, i64, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[V15:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V16:.+]] = llvm.icmp "ne" %[[V14]], %[[V15]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V16]], ^bb5, ^bb4
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:    llvm.store %[[V8]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb4:
// CHECK-NEXT:    %[[V17:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V18:.+]] = llvm.alloca %[[V17]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V19:.+]] = llvm.call @hipdnn_ep_tensor_buffer_get_gpu_ptr(%[[V4]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[V20:.+]] = llvm.call @hipdnn_ep_tensor_buffer_get_shape_ptr(%[[V4]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[V21:.+]] = llvm.addrspacecast %[[V19]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V22:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.+]] = llvm.insertvalue %[[V21]], %[[V22]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V24:.+]] = llvm.insertvalue %[[V21]], %[[V23]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V25:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V26:.+]] = llvm.insertvalue %[[V25]], %[[V24]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V27:.+]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V28:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V29:.+]] = llvm.getelementptr %[[V20]]{{\[}}%[[V28]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V30:.+]] = llvm.load %[[V29]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V31:.+]] = llvm.insertvalue %[[V30]], %[[V27]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V32:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V33:.+]] = llvm.getelementptr %[[V20]]{{\[}}%[[V32]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V34:.+]] = llvm.load %[[V33]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V35:.+]] = llvm.insertvalue %[[V34]], %[[V31]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V36:.+]] = llvm.insertvalue %[[V35]], %[[V26]][3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V37:.+]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V38:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V39:.+]] = llvm.insertvalue %[[V38]], %[[V37]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V40:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V41:.+]] = llvm.getelementptr %[[V20]]{{\[}}%[[V40]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V42:.+]] = llvm.load %[[V41]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V43:.+]] = llvm.mul %[[V38]], %[[V42]] : i64
// CHECK-NEXT:    %[[V44:.+]] = llvm.insertvalue %[[V43]], %[[V39]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V45:.+]] = llvm.insertvalue %[[V44]], %[[V36]][4] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V46:.+]] = llvm.alloca %[[V1]] x !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[V45]], %[[V46]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
// CHECK-NEXT:    %[[V47:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V48:.+]] = llvm.getelementptr %[[V18]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %[[V46]], %[[V48]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V49:.+]] = llvm.call @hipdnn_ep_tensor_buffer_get_gpu_ptr(%[[V5]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[V50:.+]] = llvm.call @hipdnn_ep_tensor_buffer_get_shape_ptr(%[[V5]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:    %[[V51:.+]] = llvm.addrspacecast %[[V49]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V52:.+]] = llvm.mlir.undef : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V53:.+]] = llvm.insertvalue %[[V51]], %[[V52]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V54:.+]] = llvm.insertvalue %[[V51]], %[[V53]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V55:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V56:.+]] = llvm.insertvalue %[[V55]], %[[V54]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V57:.+]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V58:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V59:.+]] = llvm.getelementptr %[[V50]]{{\[}}%[[V58]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V60:.+]] = llvm.load %[[V59]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V61:.+]] = llvm.insertvalue %[[V60]], %[[V57]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V62:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V63:.+]] = llvm.getelementptr %[[V50]]{{\[}}%[[V62]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V64:.+]] = llvm.load %[[V63]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V65:.+]] = llvm.insertvalue %[[V64]], %[[V61]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V66:.+]] = llvm.insertvalue %[[V65]], %[[V56]][3] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V67:.+]] = llvm.mlir.undef : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V68:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V69:.+]] = llvm.insertvalue %[[V68]], %[[V67]][1] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V70:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V71:.+]] = llvm.getelementptr %[[V50]]{{\[}}%[[V70]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V72:.+]] = llvm.load %[[V71]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V73:.+]] = llvm.mul %[[V68]], %[[V72]] : i64
// CHECK-NEXT:    %[[V74:.+]] = llvm.insertvalue %[[V73]], %[[V69]][0] : !llvm.array<2 x i64>
// CHECK-NEXT:    %[[V75:.+]] = llvm.insertvalue %[[V74]], %[[V66]][4] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V76:.+]] = llvm.alloca %[[V1]] x !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:    llvm.store %[[V75]], %[[V76]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
// CHECK-NEXT:    %[[V77:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V78:.+]] = llvm.getelementptr %[[V18]]{{\[}}%[[V77]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    llvm.store %[[V76]], %[[V78]] : !llvm.ptr, !llvm.ptr
// CHECK-NEXT:    %[[V79:.+]] = llvm.call @hipdnn_ep_state_reset_error_flag(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V80:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V81:.+]] = llvm.icmp "ne" %[[V79]], %[[V80]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V81]], ^bb7, ^bb6
// CHECK-NEXT:    ^bb5:
// CHECK-NEXT:    llvm.store %[[V14]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb6:
// CHECK-NEXT:    %[[V82:.+]] = llvm.call @main_graph(%[[ARG0]], %[[V18]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:    %[[V83:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V84:.+]] = llvm.icmp "ne" %[[V82]], %[[V83]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V84]], ^bb10, ^bb9
// CHECK-NEXT:    ^bb7:
// CHECK-NEXT:    llvm.store %[[V79]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb8:
// CHECK-NEXT:    %[[V85:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V86:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V87:.+]] = llvm.icmp "ne" %[[V85]], %[[V86]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V87]], ^bb12, ^bb11
// CHECK-NEXT:    ^bb9:
// CHECK-NEXT:    llvm.br ^bb8
// CHECK-NEXT:    ^bb10:
// CHECK-NEXT:    llvm.store %[[V82]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb11:
// CHECK-NEXT:    %[[V88:.+]] = llvm.call @hipdnn_ep_state_read_and_clear_error_flag(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V89:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V90:.+]] = llvm.icmp "ne" %[[V88]], %[[V89]] : i32
// CHECK-NEXT:    llvm.cond_br %[[V90]], ^bb14, ^bb13
// CHECK-NEXT:    ^bb12:
// CHECK-NEXT:    llvm.store %[[V85]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    ^bb13:
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[V4]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.call @hipdnn_ep_tensor_free_input(%[[ARG0]], %[[V5]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V0]] : i32
// CHECK-NEXT:    ^bb14:
// CHECK-NEXT:    llvm.store %[[V88]], %[[V2]] : i32, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @inference_cleanup(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.call @hipdnn_ep_state_cleanup(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return %[[V0]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @inference_get_metadata_json(
// CHECK-SAME:    ) -> !llvm.ptr attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.addressof @__metadata_json : !llvm.ptr
// CHECK-NEXT:    llvm.return %[[V0]] : !llvm.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:  }

func.func @main_graph(%a: tensor<?x3xf16> {onnx.name = "a"},
                      %b: tensor<?x4xf32> {onnx.name = "b"})
    -> (tensor<?x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<?x3xf16>, tensor<3x1xf16>)
      -> tensor<?x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<?x1xf16>) -> tensor<?x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<?x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<?x1xf32>, tensor<2xi64>) -> tensor<?x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<?x4xf32>, tensor<4x2xf32>)
      -> tensor<?x2xf32>
  "onnx.Return"(%y) : (tensor<?x2xf32>) -> ()
}
