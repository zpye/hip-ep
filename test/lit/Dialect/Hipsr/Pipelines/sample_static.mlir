// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on a graph where every extent is static. Each extent is a
// constant, so no allocation reads its size back out of a shape buffer.
//
// The checks cover the full LLVM IR after --hipsr-pipeline.

// RUN: hip-mlir-opt %s --onnx-dialect=modeled --hipsr-pipeline | FileCheck %s

// Two constants, then the first pool and the matmul/cast pair, then the second
// pool, the output buffer, and the expand/matmul pair.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0
// CHECK-SAME: 64>
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 32
// CHECK-SAME: 6>
// CHECK-SAME: hipdnn.input_count = 2 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 2
// CHECK-SAME: 4>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: 2
// CHECK-SAME: 3>
// CHECK-SAME: array<i64: 2
// CHECK-SAME: 4>]
// CHECK-SAME: hipdnn.num_op_state_slots = 2 : i32
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 4>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: 2
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
// CHECK-NEXT:    %[[V1:.+]] = llvm.insertvalue %[[ARG1]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.+]] = llvm.insertvalue %[[ARG2]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.+]] = llvm.insertvalue %[[ARG3]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.+]] = llvm.insertvalue %[[ARG4]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.+]] = llvm.insertvalue %[[ARG6]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.+]] = llvm.insertvalue %[[ARG5]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.+]] = llvm.insertvalue %[[ARG7]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.+]] = llvm.mlir.constant(512 : index) : i64
// CHECK-NEXT:    %[[V9:.+]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V10:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V11:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V12:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V13:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V12]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V14:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V15:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V16:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V17:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V18:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V19:.+]] = llvm.insertvalue %[[V13]], %[[V18]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V20:.+]] = llvm.insertvalue %[[V13]], %[[V19]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V21:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V22:.+]] = llvm.insertvalue %[[V21]], %[[V20]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V23:.+]] = llvm.insertvalue %[[V14]], %[[V22]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V24:.+]] = llvm.insertvalue %[[V15]], %[[V23]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V25:.+]] = llvm.insertvalue %[[V17]], %[[V24]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V26:.+]] = llvm.insertvalue %[[V16]], %[[V25]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V27:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V28:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V27]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V29:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V30:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V31:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V32:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V33:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V34:.+]] = llvm.insertvalue %[[V28]], %[[V33]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V35:.+]] = llvm.insertvalue %[[V28]], %[[V34]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V36:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V37:.+]] = llvm.insertvalue %[[V36]], %[[V35]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V38:.+]] = llvm.insertvalue %[[V29]], %[[V37]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V39:.+]] = llvm.insertvalue %[[V30]], %[[V38]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V40:.+]] = llvm.insertvalue %[[V32]], %[[V39]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V41:.+]] = llvm.insertvalue %[[V31]], %[[V40]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V42:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V43:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V44:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V45:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V46:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V47:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V48:.+]] = llvm.getelementptr %[[V47]]{{\[}}%[[V45]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V49:.+]] = llvm.ptrtoint %[[V48]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V50:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V51:.+]] = llvm.add %[[V49]], %[[V50]] : i64
// CHECK-NEXT:    %[[V52:.+]] = llvm.call @malloc(%[[V51]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V53:.+]] = llvm.ptrtoint %[[V52]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V54:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V55:.+]] = llvm.sub %[[V50]], %[[V54]] : i64
// CHECK-NEXT:    %[[V56:.+]] = llvm.add %[[V53]], %[[V55]] : i64
// CHECK-NEXT:    %[[V57:.+]] = llvm.urem %[[V56]], %[[V50]] : i64
// CHECK-NEXT:    %[[V58:.+]] = llvm.sub %[[V56]], %[[V57]] : i64
// CHECK-NEXT:    %[[V59:.+]] = llvm.inttoptr %[[V58]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V60:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V61:.+]] = llvm.insertvalue %[[V52]], %[[V60]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V62:.+]] = llvm.insertvalue %[[V59]], %[[V61]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V63:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V64:.+]] = llvm.insertvalue %[[V63]], %[[V62]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V65:.+]] = llvm.insertvalue %[[V45]], %[[V64]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V66:.+]] = llvm.insertvalue %[[V46]], %[[V65]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V67:.+]] = llvm.extractvalue %[[V66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V68:.+]] = llvm.getelementptr inbounds|nuw %[[V67]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V68]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V69:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V70:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V71:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V72:.+]] = llvm.getelementptr %[[V71]]{{\[}}%[[V69]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V73:.+]] = llvm.ptrtoint %[[V72]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V74:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V75:.+]] = llvm.add %[[V73]], %[[V74]] : i64
// CHECK-NEXT:    %[[V76:.+]] = llvm.call @malloc(%[[V75]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V77:.+]] = llvm.ptrtoint %[[V76]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V78:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V79:.+]] = llvm.sub %[[V74]], %[[V78]] : i64
// CHECK-NEXT:    %[[V80:.+]] = llvm.add %[[V77]], %[[V79]] : i64
// CHECK-NEXT:    %[[V81:.+]] = llvm.urem %[[V80]], %[[V74]] : i64
// CHECK-NEXT:    %[[V82:.+]] = llvm.sub %[[V80]], %[[V81]] : i64
// CHECK-NEXT:    %[[V83:.+]] = llvm.inttoptr %[[V82]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V84:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V85:.+]] = llvm.insertvalue %[[V76]], %[[V84]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V86:.+]] = llvm.insertvalue %[[V83]], %[[V85]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V87:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V88:.+]] = llvm.insertvalue %[[V87]], %[[V86]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V89:.+]] = llvm.insertvalue %[[V69]], %[[V88]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V90:.+]] = llvm.insertvalue %[[V70]], %[[V89]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V91:.+]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V92:.+]] = llvm.getelementptr inbounds|nuw %[[V91]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V92]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V93:.+]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V94:.+]] = llvm.getelementptr inbounds|nuw %[[V93]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V42]], %[[V94]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V95:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V96:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V97:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V98:.+]] = llvm.getelementptr %[[V97]]{{\[}}%[[V95]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V99:.+]] = llvm.ptrtoint %[[V98]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V100:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V101:.+]] = llvm.add %[[V99]], %[[V100]] : i64
// CHECK-NEXT:    %[[V102:.+]] = llvm.call @malloc(%[[V101]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V103:.+]] = llvm.ptrtoint %[[V102]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V104:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V105:.+]] = llvm.sub %[[V100]], %[[V104]] : i64
// CHECK-NEXT:    %[[V106:.+]] = llvm.add %[[V103]], %[[V105]] : i64
// CHECK-NEXT:    %[[V107:.+]] = llvm.urem %[[V106]], %[[V100]] : i64
// CHECK-NEXT:    %[[V108:.+]] = llvm.sub %[[V106]], %[[V107]] : i64
// CHECK-NEXT:    %[[V109:.+]] = llvm.inttoptr %[[V108]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V110:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V111:.+]] = llvm.insertvalue %[[V102]], %[[V110]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V112:.+]] = llvm.insertvalue %[[V109]], %[[V111]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V113:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V114:.+]] = llvm.insertvalue %[[V113]], %[[V112]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V115:.+]] = llvm.insertvalue %[[V95]], %[[V114]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V116:.+]] = llvm.insertvalue %[[V96]], %[[V115]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V117:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V118:.+]] = llvm.extractvalue %[[V90]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V119:.+]] = llvm.mul %[[V117]], %[[V118]] : i64
// CHECK-NEXT:    %[[V120:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V121:.+]] = llvm.getelementptr %[[V120]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V122:.+]] = llvm.ptrtoint %[[V121]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V123:.+]] = llvm.mul %[[V119]], %[[V122]] : i64
// CHECK-NEXT:    %[[V124:.+]] = llvm.extractvalue %[[V90]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V125:.+]] = llvm.extractvalue %[[V90]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.+]] = llvm.getelementptr %[[V124]]{{\[}}%[[V125]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V127:.+]] = llvm.extractvalue %[[V116]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V128:.+]] = llvm.extractvalue %[[V116]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V129:.+]] = llvm.getelementptr %[[V127]]{{\[}}%[[V128]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V129]], %[[V126]], %[[V123]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V130:.+]] = llvm.extractvalue %[[V90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V130]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V131:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V132:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V131]], %[[V8]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V133:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V134:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V135:.+]] = llvm.insertvalue %[[V132]], %[[V134]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V136:.+]] = llvm.insertvalue %[[V132]], %[[V135]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V137:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V138:.+]] = llvm.insertvalue %[[V137]], %[[V136]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V139:.+]] = llvm.insertvalue %[[V8]], %[[V138]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V140:.+]] = llvm.insertvalue %[[V133]], %[[V139]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V141:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V142:.+]] = llvm.extractvalue %[[V140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V143:.+]] = llvm.insertvalue %[[V142]], %[[V141]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V144:.+]] = llvm.extractvalue %[[V140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V145:.+]] = llvm.getelementptr %[[V144]]{{\[}}%[[V43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V146:.+]] = llvm.insertvalue %[[V145]], %[[V143]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V147:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V148:.+]] = llvm.insertvalue %[[V147]], %[[V146]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V149:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V150:.+]] = llvm.insertvalue %[[V149]], %[[V148]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V151:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V152:.+]] = llvm.insertvalue %[[V151]], %[[V150]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V153:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V154:.+]] = llvm.insertvalue %[[V153]], %[[V152]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V155:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V156:.+]] = llvm.insertvalue %[[V155]], %[[V154]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V157:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V158:.+]] = llvm.extractvalue %[[V140]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.+]] = llvm.insertvalue %[[V158]], %[[V157]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V160:.+]] = llvm.extractvalue %[[V140]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.+]] = llvm.getelementptr %[[V160]]{{\[}}%[[V9]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V162:.+]] = llvm.insertvalue %[[V161]], %[[V159]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V163:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V164:.+]] = llvm.insertvalue %[[V163]], %[[V162]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V165:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V166:.+]] = llvm.insertvalue %[[V165]], %[[V164]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V167:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V168:.+]] = llvm.insertvalue %[[V167]], %[[V166]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V169:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V170:.+]] = llvm.insertvalue %[[V169]], %[[V168]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V171:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.+]] = llvm.insertvalue %[[V171]], %[[V170]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V173:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V174:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V175:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V176:.+]] = llvm.getelementptr %[[V175]]{{\[}}%[[V173]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V177:.+]] = llvm.ptrtoint %[[V176]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V178:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V179:.+]] = llvm.add %[[V177]], %[[V178]] : i64
// CHECK-NEXT:    %[[V180:.+]] = llvm.call @malloc(%[[V179]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V181:.+]] = llvm.ptrtoint %[[V180]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V182:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V183:.+]] = llvm.sub %[[V178]], %[[V182]] : i64
// CHECK-NEXT:    %[[V184:.+]] = llvm.add %[[V181]], %[[V183]] : i64
// CHECK-NEXT:    %[[V185:.+]] = llvm.urem %[[V184]], %[[V178]] : i64
// CHECK-NEXT:    %[[V186:.+]] = llvm.sub %[[V184]], %[[V185]] : i64
// CHECK-NEXT:    %[[V187:.+]] = llvm.inttoptr %[[V186]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V188:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V189:.+]] = llvm.insertvalue %[[V180]], %[[V188]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V190:.+]] = llvm.insertvalue %[[V187]], %[[V189]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V191:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V192:.+]] = llvm.insertvalue %[[V191]], %[[V190]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V193:.+]] = llvm.insertvalue %[[V173]], %[[V192]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V194:.+]] = llvm.insertvalue %[[V174]], %[[V193]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V195:.+]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V196:.+]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V197:.+]] = llvm.extractvalue %[[V41]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V198:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V199:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V200:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V201:.+]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V202:.+]] = llvm.extractvalue %[[V41]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V203:.+]] = llvm.extractvalue %[[V156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V204:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V205:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V206:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V207:.+]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V200]], %[[V201]], %[[V202]], %[[V203]], %[[V195]], %[[V197]], %[[V196]], %[[V198]], %[[V204]], %[[V199]], %[[V205]], %[[V206]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V208:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V209:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V210:.+]] = llvm.mul %[[V208]], %[[V209]] : i64
// CHECK-NEXT:    %[[V211:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V212:.+]] = llvm.mul %[[V210]], %[[V211]] : i64
// CHECK-NEXT:    %[[V213:.+]] = llvm.extractvalue %[[V156]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V214:.+]] = llvm.extractvalue %[[V172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V215:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V216:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V217:.+]] = llvm.call @wrap_cast(%[[ARG0]], %[[V213]], %[[V214]], %[[V212]], %[[V215]], %[[V216]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V218:.+]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V219:.+]] = llvm.getelementptr inbounds|nuw %[[V218]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V10]], %[[V219]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V220:.+]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V221:.+]] = llvm.getelementptr inbounds|nuw %[[V220]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V11]], %[[V221]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V222:.+]] = llvm.extractvalue %[[V116]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V222]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V223:.+]] = llvm.extractvalue %[[V66]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V223]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V224:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V225:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V226:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V227:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V228:.+]] = llvm.getelementptr %[[V227]]{{\[}}%[[V225]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V229:.+]] = llvm.ptrtoint %[[V228]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V230:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V231:.+]] = llvm.add %[[V229]], %[[V230]] : i64
// CHECK-NEXT:    %[[V232:.+]] = llvm.call @malloc(%[[V231]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V233:.+]] = llvm.ptrtoint %[[V232]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V234:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V235:.+]] = llvm.sub %[[V230]], %[[V234]] : i64
// CHECK-NEXT:    %[[V236:.+]] = llvm.add %[[V233]], %[[V235]] : i64
// CHECK-NEXT:    %[[V237:.+]] = llvm.urem %[[V236]], %[[V230]] : i64
// CHECK-NEXT:    %[[V238:.+]] = llvm.sub %[[V236]], %[[V237]] : i64
// CHECK-NEXT:    %[[V239:.+]] = llvm.inttoptr %[[V238]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V240:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V241:.+]] = llvm.insertvalue %[[V232]], %[[V240]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V242:.+]] = llvm.insertvalue %[[V239]], %[[V241]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V243:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V244:.+]] = llvm.insertvalue %[[V243]], %[[V242]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V245:.+]] = llvm.insertvalue %[[V225]], %[[V244]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V246:.+]] = llvm.insertvalue %[[V226]], %[[V245]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V247:.+]] = llvm.extractvalue %[[V246]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V248:.+]] = llvm.getelementptr inbounds|nuw %[[V247]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V248]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V249:.+]] = llvm.extractvalue %[[V246]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V250:.+]] = llvm.getelementptr inbounds|nuw %[[V249]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V42]], %[[V250]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V251:.+]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V252:.+]] = llvm.getelementptr inbounds|nuw %[[V251]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V253:.+]] = llvm.load %[[V252]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V254:.+]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V255:.+]] = llvm.getelementptr inbounds|nuw %[[V254]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V256:.+]] = llvm.load %[[V255]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V257:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V258:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V259:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V260:.+]] = llvm.getelementptr %[[V259]]{{\[}}%[[V257]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V261:.+]] = llvm.ptrtoint %[[V260]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V262:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V263:.+]] = llvm.add %[[V261]], %[[V262]] : i64
// CHECK-NEXT:    %[[V264:.+]] = llvm.call @malloc(%[[V263]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V265:.+]] = llvm.ptrtoint %[[V264]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V266:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V267:.+]] = llvm.sub %[[V262]], %[[V266]] : i64
// CHECK-NEXT:    %[[V268:.+]] = llvm.add %[[V265]], %[[V267]] : i64
// CHECK-NEXT:    %[[V269:.+]] = llvm.urem %[[V268]], %[[V262]] : i64
// CHECK-NEXT:    %[[V270:.+]] = llvm.sub %[[V268]], %[[V269]] : i64
// CHECK-NEXT:    %[[V271:.+]] = llvm.inttoptr %[[V270]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V272:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V273:.+]] = llvm.insertvalue %[[V264]], %[[V272]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V274:.+]] = llvm.insertvalue %[[V271]], %[[V273]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V275:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V276:.+]] = llvm.insertvalue %[[V275]], %[[V274]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V277:.+]] = llvm.insertvalue %[[V257]], %[[V276]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V278:.+]] = llvm.insertvalue %[[V258]], %[[V277]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V279:.+]] = llvm.extractvalue %[[V278]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V280:.+]] = llvm.getelementptr inbounds|nuw %[[V279]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V253]], %[[V280]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V281:.+]] = llvm.extractvalue %[[V278]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V282:.+]] = llvm.getelementptr inbounds|nuw %[[V281]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V256]], %[[V282]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V283:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V284:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V285:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V286:.+]] = llvm.getelementptr %[[V285]]{{\[}}%[[V283]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V287:.+]] = llvm.ptrtoint %[[V286]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V288:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V289:.+]] = llvm.add %[[V287]], %[[V288]] : i64
// CHECK-NEXT:    %[[V290:.+]] = llvm.call @malloc(%[[V289]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V291:.+]] = llvm.ptrtoint %[[V290]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V292:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V293:.+]] = llvm.sub %[[V288]], %[[V292]] : i64
// CHECK-NEXT:    %[[V294:.+]] = llvm.add %[[V291]], %[[V293]] : i64
// CHECK-NEXT:    %[[V295:.+]] = llvm.urem %[[V294]], %[[V288]] : i64
// CHECK-NEXT:    %[[V296:.+]] = llvm.sub %[[V294]], %[[V295]] : i64
// CHECK-NEXT:    %[[V297:.+]] = llvm.inttoptr %[[V296]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V298:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V299:.+]] = llvm.insertvalue %[[V290]], %[[V298]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V300:.+]] = llvm.insertvalue %[[V297]], %[[V299]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V301:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V302:.+]] = llvm.insertvalue %[[V301]], %[[V300]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V303:.+]] = llvm.insertvalue %[[V283]], %[[V302]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V304:.+]] = llvm.insertvalue %[[V284]], %[[V303]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb1(%[[V43]] : i64)
// CHECK-NEXT:    ^bb1(%[[V305:.+]]: i64):
// CHECK-NEXT:    %[[V306:.+]] = llvm.icmp "slt" %[[V305]], %[[V44]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V306]], ^bb2, ^bb7
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    %[[V307:.+]] = llvm.icmp "ult" %[[V305]], %[[V43]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V307]], ^bb3, ^bb4
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:    llvm.br ^bb5(%[[V42]] : i64)
// CHECK-NEXT:    ^bb4:
// CHECK-NEXT:    %[[V308:.+]] = llvm.extractvalue %[[V246]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V309:.+]] = llvm.getelementptr inbounds|nuw %[[V308]]{{\[}}%[[V305]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V310:.+]] = llvm.load %[[V309]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V311:.+]] = llvm.extractvalue %[[V278]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V312:.+]] = llvm.getelementptr inbounds|nuw %[[V311]]{{\[}}%[[V305]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V313:.+]] = llvm.load %[[V312]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V314:.+]] = llvm.icmp "eq" %[[V313]], %[[V42]] : i64
// CHECK-NEXT:    %[[V315:.+]] = llvm.select %[[V314]], %[[V310]], %[[V313]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb5(%[[V315]] : i64)
// CHECK-NEXT:    ^bb5(%[[V316:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb6
// CHECK-NEXT:    ^bb6:
// CHECK-NEXT:    %[[V317:.+]] = llvm.extractvalue %[[V304]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V318:.+]] = llvm.getelementptr inbounds|nuw %[[V317]]{{\[}}%[[V305]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V316]], %[[V318]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V319:.+]] = llvm.add %[[V305]], %[[V42]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V319]] : i64)
// CHECK-NEXT:    ^bb7:
// CHECK-NEXT:    %[[V320:.+]] = llvm.extractvalue %[[V278]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V320]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V321:.+]] = llvm.extractvalue %[[V246]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V321]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V322:.+]] = llvm.extractvalue %[[V304]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V323:.+]] = llvm.getelementptr inbounds|nuw %[[V322]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V324:.+]] = llvm.load %[[V323]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V325:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V326:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V327:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V328:.+]] = llvm.getelementptr %[[V327]]{{\[}}%[[V325]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V329:.+]] = llvm.ptrtoint %[[V328]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V330:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V331:.+]] = llvm.add %[[V329]], %[[V330]] : i64
// CHECK-NEXT:    %[[V332:.+]] = llvm.call @malloc(%[[V331]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V333:.+]] = llvm.ptrtoint %[[V332]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V334:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V335:.+]] = llvm.sub %[[V330]], %[[V334]] : i64
// CHECK-NEXT:    %[[V336:.+]] = llvm.add %[[V333]], %[[V335]] : i64
// CHECK-NEXT:    %[[V337:.+]] = llvm.urem %[[V336]], %[[V330]] : i64
// CHECK-NEXT:    %[[V338:.+]] = llvm.sub %[[V336]], %[[V337]] : i64
// CHECK-NEXT:    %[[V339:.+]] = llvm.inttoptr %[[V338]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V340:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V341:.+]] = llvm.insertvalue %[[V332]], %[[V340]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V342:.+]] = llvm.insertvalue %[[V339]], %[[V341]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V343:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V344:.+]] = llvm.insertvalue %[[V343]], %[[V342]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V345:.+]] = llvm.insertvalue %[[V325]], %[[V344]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V346:.+]] = llvm.insertvalue %[[V326]], %[[V345]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V347:.+]] = llvm.extractvalue %[[V346]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V348:.+]] = llvm.getelementptr inbounds|nuw %[[V347]]{{\[}}%[[V43]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V324]], %[[V348]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V349:.+]] = llvm.extractvalue %[[V346]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V350:.+]] = llvm.getelementptr inbounds|nuw %[[V349]]{{\[}}%[[V42]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V44]], %[[V350]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V351:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V352:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V353:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V354:.+]] = llvm.getelementptr %[[V353]]{{\[}}%[[V351]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V355:.+]] = llvm.ptrtoint %[[V354]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V356:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V357:.+]] = llvm.add %[[V355]], %[[V356]] : i64
// CHECK-NEXT:    %[[V358:.+]] = llvm.call @malloc(%[[V357]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V359:.+]] = llvm.ptrtoint %[[V358]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V360:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V361:.+]] = llvm.sub %[[V356]], %[[V360]] : i64
// CHECK-NEXT:    %[[V362:.+]] = llvm.add %[[V359]], %[[V361]] : i64
// CHECK-NEXT:    %[[V363:.+]] = llvm.urem %[[V362]], %[[V356]] : i64
// CHECK-NEXT:    %[[V364:.+]] = llvm.sub %[[V362]], %[[V363]] : i64
// CHECK-NEXT:    %[[V365:.+]] = llvm.inttoptr %[[V364]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V366:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V367:.+]] = llvm.insertvalue %[[V358]], %[[V366]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V368:.+]] = llvm.insertvalue %[[V365]], %[[V367]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V369:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V370:.+]] = llvm.insertvalue %[[V369]], %[[V368]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V371:.+]] = llvm.insertvalue %[[V351]], %[[V370]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V372:.+]] = llvm.insertvalue %[[V352]], %[[V371]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V373:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V374:.+]] = llvm.extractvalue %[[V346]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V375:.+]] = llvm.mul %[[V373]], %[[V374]] : i64
// CHECK-NEXT:    %[[V376:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V377:.+]] = llvm.getelementptr %[[V376]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V378:.+]] = llvm.ptrtoint %[[V377]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V379:.+]] = llvm.mul %[[V375]], %[[V378]] : i64
// CHECK-NEXT:    %[[V380:.+]] = llvm.extractvalue %[[V346]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V381:.+]] = llvm.extractvalue %[[V346]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V382:.+]] = llvm.getelementptr %[[V380]]{{\[}}%[[V381]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V383:.+]] = llvm.extractvalue %[[V372]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V384:.+]] = llvm.extractvalue %[[V372]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V385:.+]] = llvm.getelementptr %[[V383]]{{\[}}%[[V384]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V385]], %[[V382]], %[[V379]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V386:.+]] = llvm.extractvalue %[[V346]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V386]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V387:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V388:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V387]], %[[V9]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V389:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V390:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V391:.+]] = llvm.insertvalue %[[V388]], %[[V390]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.+]] = llvm.insertvalue %[[V388]], %[[V391]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V393:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V394:.+]] = llvm.insertvalue %[[V393]], %[[V392]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V395:.+]] = llvm.insertvalue %[[V9]], %[[V394]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V396:.+]] = llvm.insertvalue %[[V389]], %[[V395]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V397:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V398:.+]] = llvm.extractvalue %[[V396]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V399:.+]] = llvm.insertvalue %[[V398]], %[[V397]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V400:.+]] = llvm.extractvalue %[[V396]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V401:.+]] = llvm.getelementptr %[[V400]]{{\[}}%[[V43]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V402:.+]] = llvm.insertvalue %[[V401]], %[[V399]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V403:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V404:.+]] = llvm.insertvalue %[[V403]], %[[V402]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V405:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V406:.+]] = llvm.insertvalue %[[V405]], %[[V404]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V407:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V408:.+]] = llvm.insertvalue %[[V407]], %[[V406]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V409:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V410:.+]] = llvm.insertvalue %[[V409]], %[[V408]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V411:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V412:.+]] = llvm.insertvalue %[[V411]], %[[V410]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V413:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V414:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V415:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V416:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-NEXT:    %[[V417:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V418:.+]] = llvm.getelementptr %[[V417]]{{\[}}%[[V416]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:    %[[V419:.+]] = llvm.ptrtoint %[[V418]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V420:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V421:.+]] = llvm.alloca %[[V420]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V422:.+]] = llvm.getelementptr %[[V421]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V413]], %[[V422]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V423:.+]] = llvm.getelementptr %[[V421]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V414]], %[[V423]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V424:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V425:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V426:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V427:.+]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V424]], %[[V421]], %[[V425]], %[[V426]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V428:.+]] = llvm.addrspacecast %[[V427]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V429:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V430:.+]] = llvm.insertvalue %[[V428]], %[[V429]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V431:.+]] = llvm.insertvalue %[[V428]], %[[V430]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V432:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V433:.+]] = llvm.insertvalue %[[V432]], %[[V431]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V434:.+]] = llvm.insertvalue %[[V413]], %[[V433]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V435:.+]] = llvm.insertvalue %[[V414]], %[[V434]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V436:.+]] = llvm.insertvalue %[[V414]], %[[V435]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V437:.+]] = llvm.insertvalue %[[V415]], %[[V436]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V438:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V439:.+]] = llvm.extractvalue %[[V194]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V440:.+]] = llvm.alloca %[[V438]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V441:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V442:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V443:.+]] = llvm.getelementptr %[[V440]]{{\[}}%[[V442]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V441]], %[[V443]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V444:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V445:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V446:.+]] = llvm.getelementptr %[[V440]]{{\[}}%[[V445]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V444]], %[[V446]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V447:.+]] = llvm.alloca %[[V438]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V448:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V449:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V450:.+]] = llvm.getelementptr %[[V447]]{{\[}}%[[V449]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V448]], %[[V450]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V451:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V452:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V453:.+]] = llvm.getelementptr %[[V447]]{{\[}}%[[V452]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V451]], %[[V453]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V454:.+]] = llvm.extractvalue %[[V172]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V455:.+]] = llvm.extractvalue %[[V412]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V456:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V457:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V458:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V459:.+]] = llvm.call @wrap_expand(%[[ARG0]], %[[V454]], %[[V439]], %[[V455]], %[[V440]], %[[V456]], %[[V447]], %[[V457]], %[[V458]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V460:.+]] = llvm.extractvalue %[[V194]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V460]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V461:.+]] = llvm.extractvalue %[[V412]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V462:.+]] = llvm.extractvalue %[[V412]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V463:.+]] = llvm.extractvalue %[[V26]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V464:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V465:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V466:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V467:.+]] = llvm.extractvalue %[[V412]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V468:.+]] = llvm.extractvalue %[[V26]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V469:.+]] = llvm.extractvalue %[[V437]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V470:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V471:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V472:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V473:.+]] = llvm.call @wrap_hipblasLtMatmul(%[[ARG0]], %[[V466]], %[[V467]], %[[V468]], %[[V469]], %[[V461]], %[[V463]], %[[V462]], %[[V464]], %[[V470]], %[[V465]], %[[V471]], %[[V472]]) : (!llvm.ptr, i32, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V474:.+]] = llvm.extractvalue %[[V304]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V474]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V475:.+]] = llvm.extractvalue %[[V372]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V475]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V437]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
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

func.func @main_graph(%a: tensor<2x3xf16> {onnx.name = "a"},
                      %b: tensor<2x4xf32> {onnx.name = "b"})
    -> (tensor<2x2xf32> {onnx.name = "y"})
    attributes {onnx.graph.name = "main_graph"} {
  %w1 = "onnx.Constant"() {value = dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf16>}
      : () -> tensor<3x1xf16>
  %mm1 = "onnx.MatMul"(%a, %w1) : (tensor<2x3xf16>, tensor<3x1xf16>)
      -> tensor<2x1xf16>
  %cast = "onnx.Cast"(%mm1) {to = f32} : (tensor<2x1xf16>) -> tensor<2x1xf32>
  %shape = "onnx.Shape"(%b) : (tensor<2x4xf32>) -> tensor<2xi64>
  %expand = "onnx.Expand"(%cast, %shape)
      : (tensor<2x1xf32>, tensor<2xi64>) -> tensor<2x4xf32>
  %w2 = "onnx.Constant"() {value = dense<[[1.0, 2.0], [3.0, 4.0],
                                          [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf32>}
      : () -> tensor<4x2xf32>
  %y = "onnx.MatMul"(%expand, %w2) : (tensor<2x4xf32>, tensor<4x2xf32>)
      -> tensor<2x2xf32>
  "onnx.Return"(%y) : (tensor<2x2xf32>) -> ()
}
