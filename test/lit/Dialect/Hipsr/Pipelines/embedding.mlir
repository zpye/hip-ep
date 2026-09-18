// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// The hipsr pipeline on an embedding graph with dynamic shapes. NonZero makes
// the scatter index count depend on the data, so the pipeline cuts five pool
// domains. Each domain starts with a shape computation that reads a host
// buffer an earlier domain filled.
//
// Pool domains and what cuts them
// -------------------------------
//
// A domain is one pool allocation, so every buffer inside it must be sized
// before it runs. The pipeline therefore starts a new domain wherever a shape
// depends on a value the host cannot know until the previous domain has
// finished.
//
//   domain 0   collapse(image_features)              -> flat
//              equal(input_ids, 248056) -> unsqueeze -> mask
//              gather(table, input_ids)              -> embeds
//              shape(embeds)                         -> extents  host 3xi64
//                   |
//                   |  extents: the broadcast destination is not in any type
//                   v
//   domain 1   expand(mask, extents)                 -> mask3d
//                   |
//                   |  extents: the second broadcast reads them again
//                   v
//   domain 2   expand(mask3d, extents)               -> mask3d'
//              nonzero(mask3d')                      -> coords 3x?, count
//              copy_d2h(count)                       -> count    host 1xi32
//                   |
//                   |  count: how many coordinates the search actually found
//                   v
//   domain 3   extract_slice(coords, count)          -> coords 3x?
//              transpose(coords)                     -> coords ?x3
//              shape -> gather -> unsqueeze          -> window   host 1xi64
//                   |
//                   |  window: where the update slice ends
//                   v
//   domain 4   slice(flat, window)                   -> updates
//              scatter_nd(embeds, coords, updates)   -> inputs_embeds
//
// The checks cover the full LLVM IR after --hipsr-pipeline.
//
// The embedding table lives in an external file, so the RUN line creates a
// file of the right length to map. Only the length matters; nothing reads the
// weights.

// RUN: %python %S/../../../Inputs/make_external_data.py %t/embedding.onnx.data 2034237440 && cd %t && hip-mlir-opt --onnx-dialect=modeled --hipsr-pipeline --mlir-elide-resource-strings-if-larger=32 %s | FileCheck %s

// generate-interface reads these constant-layout module attributes.
// This graph has no matmul, so the pipeline assigns no op-state slots.
// Rank-3 output: batch, sequence, hidden.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hip.constants_file = "constants.bin"
// CHECK-SAME: hipdnn.constant_offsets = array<i64: 0
// CHECK-SAME: 64>
// CHECK-SAME: hipdnn.constant_sizes = array<i64: 8
// CHECK-SAME: 2034237440>
// CHECK-SAME: hipdnn.input_count = 2 : i64
// CHECK-SAME: hipdnn.input_element_sizes = array<i64: 8
// CHECK-SAME: 2>
// CHECK-SAME: hipdnn.input_shapes = [array<i64: -1
// CHECK-SAME: -1>
// CHECK-SAME: array<i64: -1
// CHECK-SAME: 4096>]
// CHECK-SAME: hipdnn.output_count = 1 : i64
// CHECK-SAME: hipdnn.output_element_sizes = array<i64: 2>
// CHECK-SAME: hipdnn.output_shapes = [array<i64: -1
// CHECK-SAME: -1
// CHECK-SAME: 4096>]
// CHECK: llvm.mlir.global internal constant @__metadata_json
// CHECK: llvm.mlir.global internal constant @__metadata_blob
// CHECK:       llvm.func @wrap_scatter_nd(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_slice(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_alloc_output(!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @wrap_transpose(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_copy_d2h(!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_nonzero(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_expand(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_stream_sync(!llvm.ptr) -> i32
// CHECK-NEXT:  llvm.func @wrap_gather(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @wrap_equal(!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:  llvm.func @hipdnn_ep_get_pool_base(!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:  llvm.func @free(!llvm.ptr)
// CHECK-NEXT:  llvm.func @malloc(i64) -> !llvm.ptr
// CHECK-NEXT:  llvm.func @hipdnn_ep_constant_get(!llvm.ptr, i64) -> !llvm.ptr<1>
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
// CHECK-NEXT:    %[[V22:.+]] = llvm.call @main_graph_internal(%[[ARG0]], %[[V4]], %[[V5]], %[[V6]], %[[V7]], %[[V8]], %[[V9]], %[[V10]], %[[V15]], %[[V16]], %[[V17]], %[[V18]], %[[V19]], %[[V20]], %[[V21]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V23:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    llvm.return %[[V23]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func private @main_graph_internal(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr<1>, %[[ARG2:[^,]*]]: !llvm.ptr<1>, %[[ARG3:[^,]*]]: i64, %[[ARG4:[^,]*]]: i64, %[[ARG5:[^,]*]]: i64, %[[ARG6:[^,]*]]: i64, %[[ARG7:[^,]*]]: i64, %[[ARG8:[^,]*]]: !llvm.ptr<1>, %[[ARG9:[^,]*]]: !llvm.ptr<1>, %[[ARG10:[^,]*]]: i64, %[[ARG11:[^,]*]]: i64, %[[ARG12:[^,]*]]: i64, %[[ARG13:[^,]*]]: i64, %[[ARG14:[^,]*]]: i64) -> (!llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1:.+]] = llvm.insertvalue %[[ARG1]], %[[V0]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V2:.+]] = llvm.insertvalue %[[ARG2]], %[[V1]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V3:.+]] = llvm.insertvalue %[[ARG3]], %[[V2]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V4:.+]] = llvm.insertvalue %[[ARG4]], %[[V3]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V5:.+]] = llvm.insertvalue %[[ARG6]], %[[V4]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V6:.+]] = llvm.insertvalue %[[ARG5]], %[[V5]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V7:.+]] = llvm.insertvalue %[[ARG7]], %[[V6]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V8:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V9:.+]] = llvm.insertvalue %[[ARG8]], %[[V8]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V10:.+]] = llvm.insertvalue %[[ARG9]], %[[V9]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V11:.+]] = llvm.insertvalue %[[ARG10]], %[[V10]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V12:.+]] = llvm.insertvalue %[[ARG11]], %[[V11]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V13:.+]] = llvm.insertvalue %[[ARG13]], %[[V12]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V14:.+]] = llvm.insertvalue %[[ARG12]], %[[V13]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V15:.+]] = llvm.insertvalue %[[ARG14]], %[[V14]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V16:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V17:.+]] = llvm.mlir.constant(24 : index) : i64
// CHECK-NEXT:    %[[V18:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V19:.+]] = llvm.mlir.constant(8192 : index) : i64
// CHECK-NEXT:    %[[V20:.+]] = llvm.mlir.constant(255 : index) : i64
// CHECK-NEXT:    %[[V21:.+]] = llvm.mlir.constant(256 : index) : i64
// CHECK-NEXT:    %[[V22:.+]] = llvm.mlir.constant(248320 : index) : i64
// CHECK-NEXT:    %[[V23:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V24:.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V25:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V26:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V25]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V27:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V28:.+]] = llvm.insertvalue %[[V26]], %[[V27]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V29:.+]] = llvm.insertvalue %[[V26]], %[[V28]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V30:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V31:.+]] = llvm.insertvalue %[[V30]], %[[V29]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V32:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V33:.+]] = llvm.call @hipdnn_ep_constant_get(%[[ARG0]], %[[V32]]) : (!llvm.ptr, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V34:.+]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT:    %[[V35:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V36:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V37:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V38:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V39:.+]] = llvm.insertvalue %[[V33]], %[[V38]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V40:.+]] = llvm.insertvalue %[[V33]], %[[V39]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V41:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V42:.+]] = llvm.insertvalue %[[V41]], %[[V40]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V43:.+]] = llvm.insertvalue %[[V34]], %[[V42]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V44:.+]] = llvm.insertvalue %[[V35]], %[[V43]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V45:.+]] = llvm.insertvalue %[[V37]], %[[V44]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V46:.+]] = llvm.insertvalue %[[V36]], %[[V45]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V47:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V48:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V49:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V50:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V51:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V52:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V53:.+]] = llvm.getelementptr %[[V52]]{{\[}}%[[V50]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V54:.+]] = llvm.ptrtoint %[[V53]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V55:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V56:.+]] = llvm.add %[[V54]], %[[V55]] : i64
// CHECK-NEXT:    %[[V57:.+]] = llvm.call @malloc(%[[V56]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V58:.+]] = llvm.ptrtoint %[[V57]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V59:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V60:.+]] = llvm.sub %[[V55]], %[[V59]] : i64
// CHECK-NEXT:    %[[V61:.+]] = llvm.add %[[V58]], %[[V60]] : i64
// CHECK-NEXT:    %[[V62:.+]] = llvm.urem %[[V61]], %[[V55]] : i64
// CHECK-NEXT:    %[[V63:.+]] = llvm.sub %[[V61]], %[[V62]] : i64
// CHECK-NEXT:    %[[V64:.+]] = llvm.inttoptr %[[V63]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V65:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V66:.+]] = llvm.insertvalue %[[V57]], %[[V65]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V67:.+]] = llvm.insertvalue %[[V64]], %[[V66]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V68:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V69:.+]] = llvm.insertvalue %[[V68]], %[[V67]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V70:.+]] = llvm.insertvalue %[[V50]], %[[V69]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V71:.+]] = llvm.insertvalue %[[V51]], %[[V70]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V72:.+]] = llvm.extractvalue %[[V71]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V73:.+]] = llvm.getelementptr inbounds|nuw %[[V72]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V73]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V74:.+]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V75:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V76:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V77:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V78:.+]] = llvm.getelementptr %[[V77]]{{\[}}%[[V75]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V79:.+]] = llvm.ptrtoint %[[V78]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V80:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V81:.+]] = llvm.add %[[V79]], %[[V80]] : i64
// CHECK-NEXT:    %[[V82:.+]] = llvm.call @malloc(%[[V81]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V83:.+]] = llvm.ptrtoint %[[V82]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V84:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V85:.+]] = llvm.sub %[[V80]], %[[V84]] : i64
// CHECK-NEXT:    %[[V86:.+]] = llvm.add %[[V83]], %[[V85]] : i64
// CHECK-NEXT:    %[[V87:.+]] = llvm.urem %[[V86]], %[[V80]] : i64
// CHECK-NEXT:    %[[V88:.+]] = llvm.sub %[[V86]], %[[V87]] : i64
// CHECK-NEXT:    %[[V89:.+]] = llvm.inttoptr %[[V88]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V90:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V91:.+]] = llvm.insertvalue %[[V82]], %[[V90]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V92:.+]] = llvm.insertvalue %[[V89]], %[[V91]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V93:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V94:.+]] = llvm.insertvalue %[[V93]], %[[V92]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V95:.+]] = llvm.insertvalue %[[V75]], %[[V94]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V96:.+]] = llvm.insertvalue %[[V76]], %[[V95]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V97:.+]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V98:.+]] = llvm.getelementptr inbounds|nuw %[[V97]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V74]], %[[V98]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V99:.+]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V100:.+]] = llvm.getelementptr inbounds|nuw %[[V99]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V100]] : i64, !llvm.ptr
// CHECK-NEXT:    llvm.br ^bb1(%[[V48]], %[[V47]] : i64, i64)
// CHECK-NEXT:    ^bb1(%[[V101:.+]]: i64, %[[V102:.+]]: i64):
// CHECK-NEXT:    %[[V103:.+]] = llvm.icmp "slt" %[[V101]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V103]], ^bb2, ^bb3
// CHECK-NEXT:    ^bb2:
// CHECK-NEXT:    %[[V104:.+]] = llvm.extractvalue %[[V96]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V105:.+]] = llvm.getelementptr inbounds|nuw %[[V104]]{{\[}}%[[V101]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V106:.+]] = llvm.load %[[V105]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V107:.+]] = llvm.mul %[[V106]], %[[V102]] : i64
// CHECK-NEXT:    %[[V108:.+]] = llvm.add %[[V101]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb1(%[[V108]], %[[V107]] : i64, i64)
// CHECK-NEXT:    ^bb3:
// CHECK-NEXT:    %[[V109:.+]] = llvm.extractvalue %[[V96]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V110:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V111:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V112:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V113:.+]] = llvm.getelementptr %[[V112]]{{\[}}%[[V110]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V114:.+]] = llvm.ptrtoint %[[V113]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V115:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V116:.+]] = llvm.add %[[V114]], %[[V115]] : i64
// CHECK-NEXT:    %[[V117:.+]] = llvm.call @malloc(%[[V116]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V118:.+]] = llvm.ptrtoint %[[V117]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V119:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V120:.+]] = llvm.sub %[[V115]], %[[V119]] : i64
// CHECK-NEXT:    %[[V121:.+]] = llvm.add %[[V118]], %[[V120]] : i64
// CHECK-NEXT:    %[[V122:.+]] = llvm.urem %[[V121]], %[[V115]] : i64
// CHECK-NEXT:    %[[V123:.+]] = llvm.sub %[[V121]], %[[V122]] : i64
// CHECK-NEXT:    %[[V124:.+]] = llvm.inttoptr %[[V123]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V125:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V126:.+]] = llvm.insertvalue %[[V117]], %[[V125]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V127:.+]] = llvm.insertvalue %[[V124]], %[[V126]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V128:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V129:.+]] = llvm.insertvalue %[[V128]], %[[V127]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V130:.+]] = llvm.insertvalue %[[V110]], %[[V129]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V131:.+]] = llvm.insertvalue %[[V111]], %[[V130]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V132:.+]] = llvm.extractvalue %[[V131]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V133:.+]] = llvm.getelementptr inbounds|nuw %[[V132]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V102]], %[[V133]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V134:.+]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V135:.+]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V136:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V137:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V138:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V139:.+]] = llvm.getelementptr %[[V138]]{{\[}}%[[V136]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V140:.+]] = llvm.ptrtoint %[[V139]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V141:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V142:.+]] = llvm.add %[[V140]], %[[V141]] : i64
// CHECK-NEXT:    %[[V143:.+]] = llvm.call @malloc(%[[V142]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V144:.+]] = llvm.ptrtoint %[[V143]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V145:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V146:.+]] = llvm.sub %[[V141]], %[[V145]] : i64
// CHECK-NEXT:    %[[V147:.+]] = llvm.add %[[V144]], %[[V146]] : i64
// CHECK-NEXT:    %[[V148:.+]] = llvm.urem %[[V147]], %[[V141]] : i64
// CHECK-NEXT:    %[[V149:.+]] = llvm.sub %[[V147]], %[[V148]] : i64
// CHECK-NEXT:    %[[V150:.+]] = llvm.inttoptr %[[V149]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V151:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V152:.+]] = llvm.insertvalue %[[V143]], %[[V151]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V153:.+]] = llvm.insertvalue %[[V150]], %[[V152]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V154:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V155:.+]] = llvm.insertvalue %[[V154]], %[[V153]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V156:.+]] = llvm.insertvalue %[[V136]], %[[V155]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V157:.+]] = llvm.insertvalue %[[V137]], %[[V156]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V158:.+]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V159:.+]] = llvm.getelementptr inbounds|nuw %[[V158]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V134]], %[[V159]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V160:.+]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V161:.+]] = llvm.getelementptr inbounds|nuw %[[V160]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V135]], %[[V161]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V162:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V163:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V164:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V165:.+]] = llvm.getelementptr %[[V164]]{{\[}}%[[V162]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V166:.+]] = llvm.ptrtoint %[[V165]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V167:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V168:.+]] = llvm.add %[[V166]], %[[V167]] : i64
// CHECK-NEXT:    %[[V169:.+]] = llvm.call @malloc(%[[V168]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V170:.+]] = llvm.ptrtoint %[[V169]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V171:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V172:.+]] = llvm.sub %[[V167]], %[[V171]] : i64
// CHECK-NEXT:    %[[V173:.+]] = llvm.add %[[V170]], %[[V172]] : i64
// CHECK-NEXT:    %[[V174:.+]] = llvm.urem %[[V173]], %[[V167]] : i64
// CHECK-NEXT:    %[[V175:.+]] = llvm.sub %[[V173]], %[[V174]] : i64
// CHECK-NEXT:    %[[V176:.+]] = llvm.inttoptr %[[V175]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V177:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V178:.+]] = llvm.insertvalue %[[V169]], %[[V177]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V179:.+]] = llvm.insertvalue %[[V176]], %[[V178]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V180:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V181:.+]] = llvm.insertvalue %[[V180]], %[[V179]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V182:.+]] = llvm.insertvalue %[[V162]], %[[V181]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V183:.+]] = llvm.insertvalue %[[V163]], %[[V182]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V184:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V185:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V186:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V187:.+]] = llvm.getelementptr %[[V186]]{{\[}}%[[V184]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V188:.+]] = llvm.ptrtoint %[[V187]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V189:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V190:.+]] = llvm.add %[[V188]], %[[V189]] : i64
// CHECK-NEXT:    %[[V191:.+]] = llvm.call @malloc(%[[V190]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V192:.+]] = llvm.ptrtoint %[[V191]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V193:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V194:.+]] = llvm.sub %[[V189]], %[[V193]] : i64
// CHECK-NEXT:    %[[V195:.+]] = llvm.add %[[V192]], %[[V194]] : i64
// CHECK-NEXT:    %[[V196:.+]] = llvm.urem %[[V195]], %[[V189]] : i64
// CHECK-NEXT:    %[[V197:.+]] = llvm.sub %[[V195]], %[[V196]] : i64
// CHECK-NEXT:    %[[V198:.+]] = llvm.inttoptr %[[V197]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V199:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V200:.+]] = llvm.insertvalue %[[V191]], %[[V199]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V201:.+]] = llvm.insertvalue %[[V198]], %[[V200]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V202:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V203:.+]] = llvm.insertvalue %[[V202]], %[[V201]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V204:.+]] = llvm.insertvalue %[[V184]], %[[V203]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V205:.+]] = llvm.insertvalue %[[V185]], %[[V204]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb4(%[[V48]] : i64)
// CHECK-NEXT:    ^bb4(%[[V206:.+]]: i64):
// CHECK-NEXT:    %[[V207:.+]] = llvm.icmp "slt" %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V207]], ^bb5, ^bb14
// CHECK-NEXT:    ^bb5:
// CHECK-NEXT:    %[[V208:.+]] = llvm.icmp "ult" %[[V206]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V208]], ^bb6, ^bb7
// CHECK-NEXT:    ^bb6:
// CHECK-NEXT:    llvm.br ^bb8(%[[V47]] : i64)
// CHECK-NEXT:    ^bb7:
// CHECK-NEXT:    %[[V209:.+]] = llvm.extractvalue %[[V157]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V210:.+]] = llvm.getelementptr inbounds|nuw %[[V209]]{{\[}}%[[V206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V211:.+]] = llvm.load %[[V210]] : !llvm.ptr -> i64
// CHECK-NEXT:    llvm.br ^bb8(%[[V211]] : i64)
// CHECK-NEXT:    ^bb8(%[[V212:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb9
// CHECK-NEXT:    ^bb9:
// CHECK-NEXT:    %[[V213:.+]] = llvm.icmp "ult" %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V213]], ^bb10, ^bb11
// CHECK-NEXT:    ^bb10:
// CHECK-NEXT:    llvm.br ^bb12(%[[V212]] : i64)
// CHECK-NEXT:    ^bb11:
// CHECK-NEXT:    %[[V214:.+]] = llvm.sub %[[V206]], %[[V23]] : i64
// CHECK-NEXT:    %[[V215:.+]] = llvm.extractvalue %[[V183]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V216:.+]] = llvm.getelementptr inbounds|nuw %[[V215]]{{\[}}%[[V214]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V217:.+]] = llvm.load %[[V216]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V218:.+]] = llvm.icmp "eq" %[[V217]], %[[V47]] : i64
// CHECK-NEXT:    %[[V219:.+]] = llvm.select %[[V218]], %[[V212]], %[[V217]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb12(%[[V219]] : i64)
// CHECK-NEXT:    ^bb12(%[[V220:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb13
// CHECK-NEXT:    ^bb13:
// CHECK-NEXT:    %[[V221:.+]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V222:.+]] = llvm.getelementptr inbounds|nuw %[[V221]]{{\[}}%[[V206]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V220]], %[[V222]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V223:.+]] = llvm.add %[[V206]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb4(%[[V223]] : i64)
// CHECK-NEXT:    ^bb14:
// CHECK-NEXT:    %[[V224:.+]] = llvm.extractvalue %[[V183]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V224]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V225:.+]] = llvm.extractvalue %[[V157]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V225]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V226:.+]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V227:.+]] = llvm.getelementptr inbounds|nuw %[[V226]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V228:.+]] = llvm.load %[[V227]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V229:.+]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V230:.+]] = llvm.getelementptr inbounds|nuw %[[V229]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V231:.+]] = llvm.load %[[V230]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V232:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V233:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V234:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V235:.+]] = llvm.getelementptr %[[V234]]{{\[}}%[[V232]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V236:.+]] = llvm.ptrtoint %[[V235]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V237:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V238:.+]] = llvm.add %[[V236]], %[[V237]] : i64
// CHECK-NEXT:    %[[V239:.+]] = llvm.call @malloc(%[[V238]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V240:.+]] = llvm.ptrtoint %[[V239]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V241:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V242:.+]] = llvm.sub %[[V237]], %[[V241]] : i64
// CHECK-NEXT:    %[[V243:.+]] = llvm.add %[[V240]], %[[V242]] : i64
// CHECK-NEXT:    %[[V244:.+]] = llvm.urem %[[V243]], %[[V237]] : i64
// CHECK-NEXT:    %[[V245:.+]] = llvm.sub %[[V243]], %[[V244]] : i64
// CHECK-NEXT:    %[[V246:.+]] = llvm.inttoptr %[[V245]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V247:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V248:.+]] = llvm.insertvalue %[[V239]], %[[V247]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V249:.+]] = llvm.insertvalue %[[V246]], %[[V248]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V250:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V251:.+]] = llvm.insertvalue %[[V250]], %[[V249]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V252:.+]] = llvm.insertvalue %[[V232]], %[[V251]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V253:.+]] = llvm.insertvalue %[[V233]], %[[V252]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V254:.+]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V255:.+]] = llvm.getelementptr inbounds|nuw %[[V254]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V228]], %[[V255]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V256:.+]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V257:.+]] = llvm.getelementptr inbounds|nuw %[[V256]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V231]], %[[V257]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V258:.+]] = llvm.extractvalue %[[V253]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V259:.+]] = llvm.getelementptr inbounds|nuw %[[V258]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V259]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V260:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V261:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V262:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V263:.+]] = llvm.getelementptr %[[V262]]{{\[}}%[[V260]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V264:.+]] = llvm.ptrtoint %[[V263]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V265:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V266:.+]] = llvm.add %[[V264]], %[[V265]] : i64
// CHECK-NEXT:    %[[V267:.+]] = llvm.call @malloc(%[[V266]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V268:.+]] = llvm.ptrtoint %[[V267]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V269:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V270:.+]] = llvm.sub %[[V265]], %[[V269]] : i64
// CHECK-NEXT:    %[[V271:.+]] = llvm.add %[[V268]], %[[V270]] : i64
// CHECK-NEXT:    %[[V272:.+]] = llvm.urem %[[V271]], %[[V265]] : i64
// CHECK-NEXT:    %[[V273:.+]] = llvm.sub %[[V271]], %[[V272]] : i64
// CHECK-NEXT:    %[[V274:.+]] = llvm.inttoptr %[[V273]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V275:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V276:.+]] = llvm.insertvalue %[[V267]], %[[V275]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V277:.+]] = llvm.insertvalue %[[V274]], %[[V276]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V278:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V279:.+]] = llvm.insertvalue %[[V278]], %[[V277]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V280:.+]] = llvm.insertvalue %[[V260]], %[[V279]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V281:.+]] = llvm.insertvalue %[[V261]], %[[V280]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V282:.+]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V283:.+]] = llvm.getelementptr inbounds|nuw %[[V282]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V22]], %[[V283]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V284:.+]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V285:.+]] = llvm.getelementptr inbounds|nuw %[[V284]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V285]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V286:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V287:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V288:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V289:.+]] = llvm.getelementptr %[[V288]]{{\[}}%[[V286]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V290:.+]] = llvm.ptrtoint %[[V289]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V291:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V292:.+]] = llvm.add %[[V290]], %[[V291]] : i64
// CHECK-NEXT:    %[[V293:.+]] = llvm.call @malloc(%[[V292]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V294:.+]] = llvm.ptrtoint %[[V293]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V295:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V296:.+]] = llvm.sub %[[V291]], %[[V295]] : i64
// CHECK-NEXT:    %[[V297:.+]] = llvm.add %[[V294]], %[[V296]] : i64
// CHECK-NEXT:    %[[V298:.+]] = llvm.urem %[[V297]], %[[V291]] : i64
// CHECK-NEXT:    %[[V299:.+]] = llvm.sub %[[V297]], %[[V298]] : i64
// CHECK-NEXT:    %[[V300:.+]] = llvm.inttoptr %[[V299]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V301:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V302:.+]] = llvm.insertvalue %[[V293]], %[[V301]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V303:.+]] = llvm.insertvalue %[[V300]], %[[V302]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V304:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V305:.+]] = llvm.insertvalue %[[V304]], %[[V303]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V306:.+]] = llvm.insertvalue %[[V286]], %[[V305]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V307:.+]] = llvm.insertvalue %[[V287]], %[[V306]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V308:.+]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V309:.+]] = llvm.getelementptr inbounds|nuw %[[V308]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V134]], %[[V309]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V310:.+]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V311:.+]] = llvm.getelementptr inbounds|nuw %[[V310]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V135]], %[[V311]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V312:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V313:.+]] = llvm.extractvalue %[[V281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V314:.+]] = llvm.extractvalue %[[V281]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V315:.+]] = llvm.insertvalue %[[V313]], %[[V312]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V316:.+]] = llvm.insertvalue %[[V314]], %[[V315]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V317:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V318:.+]] = llvm.insertvalue %[[V317]], %[[V316]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V319:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V320:.+]] = llvm.insertvalue %[[V319]], %[[V318]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V321:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V322:.+]] = llvm.insertvalue %[[V321]], %[[V320]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V323:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V324:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V325:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V326:.+]] = llvm.getelementptr %[[V325]]{{\[}}%[[V323]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V327:.+]] = llvm.ptrtoint %[[V326]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V328:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V329:.+]] = llvm.add %[[V327]], %[[V328]] : i64
// CHECK-NEXT:    %[[V330:.+]] = llvm.call @malloc(%[[V329]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V331:.+]] = llvm.ptrtoint %[[V330]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V332:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V333:.+]] = llvm.sub %[[V328]], %[[V332]] : i64
// CHECK-NEXT:    %[[V334:.+]] = llvm.add %[[V331]], %[[V333]] : i64
// CHECK-NEXT:    %[[V335:.+]] = llvm.urem %[[V334]], %[[V328]] : i64
// CHECK-NEXT:    %[[V336:.+]] = llvm.sub %[[V334]], %[[V335]] : i64
// CHECK-NEXT:    %[[V337:.+]] = llvm.inttoptr %[[V336]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V338:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V339:.+]] = llvm.insertvalue %[[V330]], %[[V338]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V340:.+]] = llvm.insertvalue %[[V337]], %[[V339]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V341:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V342:.+]] = llvm.insertvalue %[[V341]], %[[V340]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V343:.+]] = llvm.insertvalue %[[V323]], %[[V342]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V344:.+]] = llvm.insertvalue %[[V324]], %[[V343]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V345:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V346:.+]] = llvm.extractvalue %[[V307]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V347:.+]] = llvm.mul %[[V345]], %[[V346]] : i64
// CHECK-NEXT:    %[[V348:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V349:.+]] = llvm.getelementptr %[[V348]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V350:.+]] = llvm.ptrtoint %[[V349]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V351:.+]] = llvm.mul %[[V347]], %[[V350]] : i64
// CHECK-NEXT:    %[[V352:.+]] = llvm.extractvalue %[[V307]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V353:.+]] = llvm.extractvalue %[[V307]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V354:.+]] = llvm.getelementptr %[[V352]]{{\[}}%[[V353]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V355:.+]] = llvm.extractvalue %[[V344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V356:.+]] = llvm.extractvalue %[[V344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V357:.+]] = llvm.getelementptr %[[V355]]{{\[}}%[[V356]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V357]], %[[V354]], %[[V351]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V358:.+]] = llvm.extractvalue %[[V307]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V358]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V359:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V360:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V361:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V362:.+]] = llvm.getelementptr %[[V361]]{{\[}}%[[V359]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V363:.+]] = llvm.ptrtoint %[[V362]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V364:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V365:.+]] = llvm.add %[[V363]], %[[V364]] : i64
// CHECK-NEXT:    %[[V366:.+]] = llvm.call @malloc(%[[V365]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V367:.+]] = llvm.ptrtoint %[[V366]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V368:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V369:.+]] = llvm.sub %[[V364]], %[[V368]] : i64
// CHECK-NEXT:    %[[V370:.+]] = llvm.add %[[V367]], %[[V369]] : i64
// CHECK-NEXT:    %[[V371:.+]] = llvm.urem %[[V370]], %[[V364]] : i64
// CHECK-NEXT:    %[[V372:.+]] = llvm.sub %[[V370]], %[[V371]] : i64
// CHECK-NEXT:    %[[V373:.+]] = llvm.inttoptr %[[V372]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V374:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V375:.+]] = llvm.insertvalue %[[V366]], %[[V374]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V376:.+]] = llvm.insertvalue %[[V373]], %[[V375]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V377:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V378:.+]] = llvm.insertvalue %[[V377]], %[[V376]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V379:.+]] = llvm.insertvalue %[[V359]], %[[V378]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V380:.+]] = llvm.insertvalue %[[V360]], %[[V379]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V381:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V382:.+]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V383:.+]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V384:.+]] = llvm.insertvalue %[[V382]], %[[V381]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V385:.+]] = llvm.insertvalue %[[V383]], %[[V384]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V386:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V387:.+]] = llvm.insertvalue %[[V386]], %[[V385]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V388:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V389:.+]] = llvm.insertvalue %[[V388]], %[[V387]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V390:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V391:.+]] = llvm.insertvalue %[[V390]], %[[V389]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V392:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V393:.+]] = llvm.extractvalue %[[V344]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V394:.+]] = llvm.mul %[[V392]], %[[V393]] : i64
// CHECK-NEXT:    %[[V395:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V396:.+]] = llvm.getelementptr %[[V395]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V397:.+]] = llvm.ptrtoint %[[V396]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V398:.+]] = llvm.mul %[[V394]], %[[V397]] : i64
// CHECK-NEXT:    %[[V399:.+]] = llvm.extractvalue %[[V344]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V400:.+]] = llvm.extractvalue %[[V344]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V401:.+]] = llvm.getelementptr %[[V399]]{{\[}}%[[V400]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V402:.+]] = llvm.extractvalue %[[V391]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V403:.+]] = llvm.extractvalue %[[V391]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V404:.+]] = llvm.getelementptr %[[V402]]{{\[}}%[[V403]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V404]], %[[V401]], %[[V398]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V405:.+]] = llvm.extractvalue %[[V344]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V405]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V406:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V407:.+]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V408:.+]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V409:.+]] = llvm.insertvalue %[[V407]], %[[V406]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V410:.+]] = llvm.insertvalue %[[V408]], %[[V409]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V411:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V412:.+]] = llvm.insertvalue %[[V411]], %[[V410]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V413:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V414:.+]] = llvm.insertvalue %[[V413]], %[[V412]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V415:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V416:.+]] = llvm.insertvalue %[[V415]], %[[V414]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V417:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V418:.+]] = llvm.extractvalue %[[V322]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V419:.+]] = llvm.mul %[[V417]], %[[V418]] : i64
// CHECK-NEXT:    %[[V420:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V421:.+]] = llvm.getelementptr %[[V420]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V422:.+]] = llvm.ptrtoint %[[V421]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V423:.+]] = llvm.mul %[[V419]], %[[V422]] : i64
// CHECK-NEXT:    %[[V424:.+]] = llvm.extractvalue %[[V322]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V425:.+]] = llvm.extractvalue %[[V322]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V426:.+]] = llvm.getelementptr %[[V424]]{{\[}}%[[V425]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V427:.+]] = llvm.extractvalue %[[V416]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V428:.+]] = llvm.extractvalue %[[V416]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V429:.+]] = llvm.getelementptr %[[V427]]{{\[}}%[[V428]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    "llvm.intr.memcpy"(%[[V429]], %[[V426]], %[[V423]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:    %[[V430:.+]] = llvm.extractvalue %[[V281]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V430]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V431:.+]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V432:.+]] = llvm.getelementptr inbounds|nuw %[[V431]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V433:.+]] = llvm.load %[[V432]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V434:.+]] = llvm.extractvalue %[[V205]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V435:.+]] = llvm.getelementptr inbounds|nuw %[[V434]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V436:.+]] = llvm.load %[[V435]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V437:.+]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V438:.+]] = llvm.getelementptr inbounds|nuw %[[V437]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V439:.+]] = llvm.load %[[V438]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V440:.+]] = llvm.extractvalue %[[V380]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V441:.+]] = llvm.getelementptr inbounds|nuw %[[V440]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V442:.+]] = llvm.load %[[V441]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V443:.+]] = llvm.mul %[[V433]], %[[V436]] : i64
// CHECK-NEXT:    %[[V444:.+]] = llvm.add %[[V443]], %[[V20]] : i64
// CHECK-NEXT:    %[[V445:.+]] = llvm.udiv %[[V444]], %[[V21]] : i64
// CHECK-NEXT:    %[[V446:.+]] = llvm.mul %[[V445]], %[[V21]] : i64
// CHECK-NEXT:    %[[V447:.+]] = llvm.mul %[[V439]], %[[V19]] : i64
// CHECK-NEXT:    %[[V448:.+]] = llvm.mul %[[V447]], %[[V442]] : i64
// CHECK-NEXT:    %[[V449:.+]] = llvm.add %[[V448]], %[[V20]] : i64
// CHECK-NEXT:    %[[V450:.+]] = llvm.udiv %[[V449]], %[[V21]] : i64
// CHECK-NEXT:    %[[V451:.+]] = llvm.mul %[[V450]], %[[V21]] : i64
// CHECK-NEXT:    %[[V452:.+]] = llvm.add %[[V446]], %[[V451]] : i64
// CHECK-NEXT:    %[[V453:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V454:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V453]], %[[V452]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V455:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V456:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V457:.+]] = llvm.insertvalue %[[V454]], %[[V456]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V458:.+]] = llvm.insertvalue %[[V454]], %[[V457]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V459:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V460:.+]] = llvm.insertvalue %[[V459]], %[[V458]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V461:.+]] = llvm.insertvalue %[[V452]], %[[V460]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V462:.+]] = llvm.insertvalue %[[V455]], %[[V461]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V463:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V464:.+]] = llvm.extractvalue %[[V462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V465:.+]] = llvm.insertvalue %[[V464]], %[[V463]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V466:.+]] = llvm.extractvalue %[[V462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V467:.+]] = llvm.getelementptr %[[V466]]{{\[}}%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V468:.+]] = llvm.insertvalue %[[V467]], %[[V465]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V469:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V470:.+]] = llvm.insertvalue %[[V469]], %[[V468]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V471:.+]] = llvm.insertvalue %[[V436]], %[[V470]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V472:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V473:.+]] = llvm.insertvalue %[[V472]], %[[V471]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V474:.+]] = llvm.insertvalue %[[V433]], %[[V473]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V475:.+]] = llvm.mul %[[V472]], %[[V436]] : i64
// CHECK-NEXT:    %[[V476:.+]] = llvm.insertvalue %[[V475]], %[[V474]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V477:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V478:.+]] = llvm.extractvalue %[[V462]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V479:.+]] = llvm.insertvalue %[[V478]], %[[V477]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V480:.+]] = llvm.extractvalue %[[V462]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V481:.+]] = llvm.getelementptr %[[V480]]{{\[}}%[[V446]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V482:.+]] = llvm.insertvalue %[[V481]], %[[V479]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V483:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V484:.+]] = llvm.insertvalue %[[V483]], %[[V482]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V485:.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V486:.+]] = llvm.insertvalue %[[V485]], %[[V484]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V487:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V488:.+]] = llvm.insertvalue %[[V487]], %[[V486]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V489:.+]] = llvm.insertvalue %[[V442]], %[[V488]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V490:.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V491:.+]] = llvm.insertvalue %[[V490]], %[[V489]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V492:.+]] = llvm.insertvalue %[[V439]], %[[V491]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V493:.+]] = llvm.mul %[[V490]], %[[V442]] : i64
// CHECK-NEXT:    %[[V494:.+]] = llvm.insertvalue %[[V493]], %[[V492]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V495:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V496:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V497:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V498:.+]] = llvm.getelementptr %[[V497]]{{\[}}%[[V495]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V499:.+]] = llvm.ptrtoint %[[V498]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V500:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V501:.+]] = llvm.add %[[V499]], %[[V500]] : i64
// CHECK-NEXT:    %[[V502:.+]] = llvm.call @malloc(%[[V501]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V503:.+]] = llvm.ptrtoint %[[V502]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V504:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V505:.+]] = llvm.sub %[[V500]], %[[V504]] : i64
// CHECK-NEXT:    %[[V506:.+]] = llvm.add %[[V503]], %[[V505]] : i64
// CHECK-NEXT:    %[[V507:.+]] = llvm.urem %[[V506]], %[[V500]] : i64
// CHECK-NEXT:    %[[V508:.+]] = llvm.sub %[[V506]], %[[V507]] : i64
// CHECK-NEXT:    %[[V509:.+]] = llvm.inttoptr %[[V508]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V510:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V511:.+]] = llvm.insertvalue %[[V502]], %[[V510]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V512:.+]] = llvm.insertvalue %[[V509]], %[[V511]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V513:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V514:.+]] = llvm.insertvalue %[[V513]], %[[V512]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V515:.+]] = llvm.insertvalue %[[V495]], %[[V514]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V516:.+]] = llvm.insertvalue %[[V496]], %[[V515]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V517:.+]] = llvm.extractvalue %[[V15]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V518:.+]] = llvm.extractvalue %[[V15]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V519:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V520:.+]] = llvm.insertvalue %[[V517]], %[[V519]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V521:.+]] = llvm.insertvalue %[[V518]], %[[V520]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V522:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V523:.+]] = llvm.insertvalue %[[V522]], %[[V521]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V524:.+]] = llvm.extractvalue %[[V15]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V525:.+]] = llvm.extractvalue %[[V15]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V526:.+]] = llvm.extractvalue %[[V15]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V527:.+]] = llvm.extractvalue %[[V15]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V528:.+]] = llvm.extractvalue %[[V15]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V529:.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V530:.+]] = llvm.mul %[[V525]], %[[V529]] overflow<nsw> : i64
// CHECK-NEXT:    %[[V531:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V532:.+]] = llvm.extractvalue %[[V523]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V533:.+]] = llvm.extractvalue %[[V523]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V534:.+]] = llvm.insertvalue %[[V532]], %[[V531]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V535:.+]] = llvm.insertvalue %[[V533]], %[[V534]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V536:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V537:.+]] = llvm.insertvalue %[[V536]], %[[V535]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V538:.+]] = llvm.insertvalue %[[V530]], %[[V537]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V539:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V540:.+]] = llvm.insertvalue %[[V539]], %[[V538]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V541:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V542:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V543:.+]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V544:.+]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V545:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V546:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V547:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V548:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V549:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V550:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V551:.+]] = llvm.extractvalue %[[V476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V552:.+]] = llvm.extractvalue %[[V476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V553:.+]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V554:.+]] = llvm.extractvalue %[[V31]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V555:.+]] = llvm.extractvalue %[[V476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V556:.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-NEXT:    %[[V557:.+]] = llvm.call @wrap_equal(%[[ARG0]], %[[V553]], %[[V554]], %[[V555]], %[[V541]], %[[V542]], %[[V543]], %[[V544]], %[[V545]], %[[V546]], %[[V547]], %[[V548]], %[[V549]], %[[V550]], %[[V551]], %[[V552]], %[[V556]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V558:.+]] = llvm.extractvalue %[[V476]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V559:.+]] = llvm.extractvalue %[[V476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V560:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V561:.+]] = llvm.insertvalue %[[V558]], %[[V560]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V562:.+]] = llvm.insertvalue %[[V559]], %[[V561]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V563:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V564:.+]] = llvm.insertvalue %[[V563]], %[[V562]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V565:.+]] = llvm.extractvalue %[[V476]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V566:.+]] = llvm.extractvalue %[[V476]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V567:.+]] = llvm.extractvalue %[[V476]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V568:.+]] = llvm.extractvalue %[[V476]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V569:.+]] = llvm.extractvalue %[[V476]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V570:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V571:.+]] = llvm.extractvalue %[[V564]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V572:.+]] = llvm.extractvalue %[[V564]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V573:.+]] = llvm.insertvalue %[[V571]], %[[V570]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V574:.+]] = llvm.insertvalue %[[V572]], %[[V573]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V575:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V576:.+]] = llvm.insertvalue %[[V575]], %[[V574]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V577:.+]] = llvm.insertvalue %[[V566]], %[[V576]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V578:.+]] = llvm.insertvalue %[[V568]], %[[V577]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V579:.+]] = llvm.insertvalue %[[V567]], %[[V578]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V580:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V581:.+]] = llvm.insertvalue %[[V580]], %[[V579]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V582:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V583:.+]] = llvm.insertvalue %[[V582]], %[[V581]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V584:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V585:.+]] = llvm.insertvalue %[[V584]], %[[V583]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V586:.+]] = llvm.mlir.constant(248320 : i64) : i64
// CHECK-NEXT:    %[[V587:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V588:.+]] = llvm.extractvalue %[[V7]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V589:.+]] = llvm.extractvalue %[[V7]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V590:.+]] = llvm.extractvalue %[[V494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V591:.+]] = llvm.extractvalue %[[V494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V592:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V593:.+]] = llvm.mul %[[V586]], %[[V587]] : i64
// CHECK-NEXT:    %[[V594:.+]] = llvm.mul %[[V590]], %[[V591]] : i64
// CHECK-NEXT:    %[[V595:.+]] = llvm.mul %[[V594]], %[[V592]] : i64
// CHECK-NEXT:    %[[V596:.+]] = llvm.mul %[[V588]], %[[V589]] : i64
// CHECK-NEXT:    %[[V597:.+]] = llvm.extractvalue %[[V46]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V598:.+]] = llvm.extractvalue %[[V7]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V599:.+]] = llvm.extractvalue %[[V494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V600:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V601:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V602:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V603:.+]] = llvm.call @wrap_gather(%[[ARG0]], %[[V597]], %[[V598]], %[[V599]], %[[V600]], %[[V593]], %[[V596]], %[[V595]], %[[V586]], %[[V587]], %[[V601]], %[[V602]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, i64, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V604:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V605:.+]] = llvm.getelementptr inbounds|nuw %[[V604]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V439]], %[[V605]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V606:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V607:.+]] = llvm.getelementptr inbounds|nuw %[[V606]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V607]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V608:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V609:.+]] = llvm.getelementptr inbounds|nuw %[[V608]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V18]], %[[V609]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V610:.+]] = llvm.extractvalue %[[V131]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V610]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V611:.+]] = llvm.extractvalue %[[V205]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V611]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V612:.+]] = llvm.extractvalue %[[V253]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V612]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V613:.+]] = llvm.extractvalue %[[V380]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V613]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V614:.+]] = llvm.extractvalue %[[V71]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V614]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V615:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V616:.+]] = llvm.extractvalue %[[V585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V617:.+]] = llvm.extractvalue %[[V585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V618:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V619:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V620:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V621:.+]] = llvm.getelementptr %[[V620]]{{\[}}%[[V618]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V622:.+]] = llvm.ptrtoint %[[V621]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V623:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V624:.+]] = llvm.add %[[V622]], %[[V623]] : i64
// CHECK-NEXT:    %[[V625:.+]] = llvm.call @malloc(%[[V624]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V626:.+]] = llvm.ptrtoint %[[V625]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V627:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V628:.+]] = llvm.sub %[[V623]], %[[V627]] : i64
// CHECK-NEXT:    %[[V629:.+]] = llvm.add %[[V626]], %[[V628]] : i64
// CHECK-NEXT:    %[[V630:.+]] = llvm.urem %[[V629]], %[[V623]] : i64
// CHECK-NEXT:    %[[V631:.+]] = llvm.sub %[[V629]], %[[V630]] : i64
// CHECK-NEXT:    %[[V632:.+]] = llvm.inttoptr %[[V631]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V633:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V634:.+]] = llvm.insertvalue %[[V625]], %[[V633]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V635:.+]] = llvm.insertvalue %[[V632]], %[[V634]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V636:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V637:.+]] = llvm.insertvalue %[[V636]], %[[V635]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V638:.+]] = llvm.insertvalue %[[V618]], %[[V637]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V639:.+]] = llvm.insertvalue %[[V619]], %[[V638]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V640:.+]] = llvm.extractvalue %[[V639]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V641:.+]] = llvm.getelementptr inbounds|nuw %[[V640]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V616]], %[[V641]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V642:.+]] = llvm.extractvalue %[[V639]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V643:.+]] = llvm.getelementptr inbounds|nuw %[[V642]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V617]], %[[V643]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V644:.+]] = llvm.extractvalue %[[V639]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V645:.+]] = llvm.getelementptr inbounds|nuw %[[V644]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V645]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V646:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V647:.+]] = llvm.getelementptr inbounds|nuw %[[V646]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V648:.+]] = llvm.load %[[V647]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V649:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V650:.+]] = llvm.getelementptr inbounds|nuw %[[V649]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V651:.+]] = llvm.load %[[V650]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V652:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V653:.+]] = llvm.getelementptr inbounds|nuw %[[V652]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V654:.+]] = llvm.load %[[V653]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V655:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V656:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V657:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V658:.+]] = llvm.getelementptr %[[V657]]{{\[}}%[[V655]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V659:.+]] = llvm.ptrtoint %[[V658]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V660:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V661:.+]] = llvm.add %[[V659]], %[[V660]] : i64
// CHECK-NEXT:    %[[V662:.+]] = llvm.call @malloc(%[[V661]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V663:.+]] = llvm.ptrtoint %[[V662]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V664:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V665:.+]] = llvm.sub %[[V660]], %[[V664]] : i64
// CHECK-NEXT:    %[[V666:.+]] = llvm.add %[[V663]], %[[V665]] : i64
// CHECK-NEXT:    %[[V667:.+]] = llvm.urem %[[V666]], %[[V660]] : i64
// CHECK-NEXT:    %[[V668:.+]] = llvm.sub %[[V666]], %[[V667]] : i64
// CHECK-NEXT:    %[[V669:.+]] = llvm.inttoptr %[[V668]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V670:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V671:.+]] = llvm.insertvalue %[[V662]], %[[V670]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V672:.+]] = llvm.insertvalue %[[V669]], %[[V671]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V673:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V674:.+]] = llvm.insertvalue %[[V673]], %[[V672]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V675:.+]] = llvm.insertvalue %[[V655]], %[[V674]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V676:.+]] = llvm.insertvalue %[[V656]], %[[V675]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V677:.+]] = llvm.extractvalue %[[V676]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V678:.+]] = llvm.getelementptr inbounds|nuw %[[V677]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V648]], %[[V678]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V679:.+]] = llvm.extractvalue %[[V676]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V680:.+]] = llvm.getelementptr inbounds|nuw %[[V679]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V651]], %[[V680]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V681:.+]] = llvm.extractvalue %[[V676]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V682:.+]] = llvm.getelementptr inbounds|nuw %[[V681]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V654]], %[[V682]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V683:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V684:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V685:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V686:.+]] = llvm.getelementptr %[[V685]]{{\[}}%[[V683]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V687:.+]] = llvm.ptrtoint %[[V686]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V688:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V689:.+]] = llvm.add %[[V687]], %[[V688]] : i64
// CHECK-NEXT:    %[[V690:.+]] = llvm.call @malloc(%[[V689]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V691:.+]] = llvm.ptrtoint %[[V690]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V692:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V693:.+]] = llvm.sub %[[V688]], %[[V692]] : i64
// CHECK-NEXT:    %[[V694:.+]] = llvm.add %[[V691]], %[[V693]] : i64
// CHECK-NEXT:    %[[V695:.+]] = llvm.urem %[[V694]], %[[V688]] : i64
// CHECK-NEXT:    %[[V696:.+]] = llvm.sub %[[V694]], %[[V695]] : i64
// CHECK-NEXT:    %[[V697:.+]] = llvm.inttoptr %[[V696]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V698:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V699:.+]] = llvm.insertvalue %[[V690]], %[[V698]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V700:.+]] = llvm.insertvalue %[[V697]], %[[V699]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V701:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V702:.+]] = llvm.insertvalue %[[V701]], %[[V700]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V703:.+]] = llvm.insertvalue %[[V683]], %[[V702]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V704:.+]] = llvm.insertvalue %[[V684]], %[[V703]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb15(%[[V48]] : i64)
// CHECK-NEXT:    ^bb15(%[[V705:.+]]: i64):
// CHECK-NEXT:    %[[V706:.+]] = llvm.icmp "slt" %[[V705]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V706]], ^bb16, ^bb21
// CHECK-NEXT:    ^bb16:
// CHECK-NEXT:    %[[V707:.+]] = llvm.icmp "ult" %[[V705]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V707]], ^bb17, ^bb18
// CHECK-NEXT:    ^bb17:
// CHECK-NEXT:    llvm.br ^bb19(%[[V47]] : i64)
// CHECK-NEXT:    ^bb18:
// CHECK-NEXT:    %[[V708:.+]] = llvm.extractvalue %[[V639]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V709:.+]] = llvm.getelementptr inbounds|nuw %[[V708]]{{\[}}%[[V705]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V710:.+]] = llvm.load %[[V709]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V711:.+]] = llvm.extractvalue %[[V676]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V712:.+]] = llvm.getelementptr inbounds|nuw %[[V711]]{{\[}}%[[V705]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V713:.+]] = llvm.load %[[V712]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V714:.+]] = llvm.icmp "eq" %[[V713]], %[[V47]] : i64
// CHECK-NEXT:    %[[V715:.+]] = llvm.select %[[V714]], %[[V710]], %[[V713]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb19(%[[V715]] : i64)
// CHECK-NEXT:    ^bb19(%[[V716:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb20
// CHECK-NEXT:    ^bb20:
// CHECK-NEXT:    %[[V717:.+]] = llvm.extractvalue %[[V704]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V718:.+]] = llvm.getelementptr inbounds|nuw %[[V717]]{{\[}}%[[V705]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V716]], %[[V718]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V719:.+]] = llvm.add %[[V705]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb15(%[[V719]] : i64)
// CHECK-NEXT:    ^bb21:
// CHECK-NEXT:    %[[V720:.+]] = llvm.extractvalue %[[V676]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V720]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V721:.+]] = llvm.extractvalue %[[V639]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V721]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V722:.+]] = llvm.extractvalue %[[V704]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V723:.+]] = llvm.getelementptr inbounds|nuw %[[V722]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V724:.+]] = llvm.load %[[V723]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V725:.+]] = llvm.extractvalue %[[V704]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V726:.+]] = llvm.getelementptr inbounds|nuw %[[V725]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V727:.+]] = llvm.load %[[V726]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V728:.+]] = llvm.extractvalue %[[V704]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V729:.+]] = llvm.getelementptr inbounds|nuw %[[V728]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V730:.+]] = llvm.load %[[V729]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V731:.+]] = llvm.mul %[[V724]], %[[V727]] : i64
// CHECK-NEXT:    %[[V732:.+]] = llvm.mul %[[V731]], %[[V730]] : i64
// CHECK-NEXT:    %[[V733:.+]] = llvm.add %[[V732]], %[[V20]] : i64
// CHECK-NEXT:    %[[V734:.+]] = llvm.udiv %[[V733]], %[[V21]] : i64
// CHECK-NEXT:    %[[V735:.+]] = llvm.mul %[[V734]], %[[V21]] : i64
// CHECK-NEXT:    %[[V736:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V737:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V736]], %[[V735]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V738:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V739:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V740:.+]] = llvm.insertvalue %[[V737]], %[[V739]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V741:.+]] = llvm.insertvalue %[[V737]], %[[V740]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V742:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V743:.+]] = llvm.insertvalue %[[V742]], %[[V741]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V744:.+]] = llvm.insertvalue %[[V735]], %[[V743]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V745:.+]] = llvm.insertvalue %[[V738]], %[[V744]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V746:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V747:.+]] = llvm.extractvalue %[[V745]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V748:.+]] = llvm.insertvalue %[[V747]], %[[V746]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V749:.+]] = llvm.extractvalue %[[V745]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V750:.+]] = llvm.getelementptr %[[V749]]{{\[}}%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V751:.+]] = llvm.insertvalue %[[V750]], %[[V748]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V752:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V753:.+]] = llvm.insertvalue %[[V752]], %[[V751]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V754:.+]] = llvm.insertvalue %[[V730]], %[[V753]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V755:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V756:.+]] = llvm.insertvalue %[[V755]], %[[V754]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V757:.+]] = llvm.insertvalue %[[V727]], %[[V756]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V758:.+]] = llvm.mul %[[V755]], %[[V730]] : i64
// CHECK-NEXT:    %[[V759:.+]] = llvm.insertvalue %[[V758]], %[[V757]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V760:.+]] = llvm.insertvalue %[[V724]], %[[V759]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V761:.+]] = llvm.mul %[[V758]], %[[V727]] : i64
// CHECK-NEXT:    %[[V762:.+]] = llvm.insertvalue %[[V761]], %[[V760]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V763:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V764:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V765:.+]] = llvm.alloca %[[V763]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V766:.+]] = llvm.extractvalue %[[V585]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V767:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V768:.+]] = llvm.getelementptr %[[V765]]{{\[}}%[[V767]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V766]], %[[V768]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V769:.+]] = llvm.extractvalue %[[V585]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V770:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V771:.+]] = llvm.getelementptr %[[V765]]{{\[}}%[[V770]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V769]], %[[V771]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V772:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V773:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V774:.+]] = llvm.getelementptr %[[V765]]{{\[}}%[[V773]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V772]], %[[V774]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V775:.+]] = llvm.alloca %[[V763]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V776:.+]] = llvm.extractvalue %[[V762]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V777:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V778:.+]] = llvm.getelementptr %[[V775]]{{\[}}%[[V777]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V776]], %[[V778]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V779:.+]] = llvm.extractvalue %[[V762]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V780:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V781:.+]] = llvm.getelementptr %[[V775]]{{\[}}%[[V780]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V779]], %[[V781]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V782:.+]] = llvm.extractvalue %[[V762]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V783:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V784:.+]] = llvm.getelementptr %[[V775]]{{\[}}%[[V783]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V782]], %[[V784]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V785:.+]] = llvm.extractvalue %[[V585]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V786:.+]] = llvm.extractvalue %[[V762]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V787:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V788:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V789:.+]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V790:.+]] = llvm.call @wrap_expand(%[[ARG0]], %[[V785]], %[[V764]], %[[V786]], %[[V765]], %[[V787]], %[[V775]], %[[V788]], %[[V789]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V791:.+]] = llvm.extractvalue %[[V704]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V791]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V792:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V793:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V794:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V795:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V796:.+]] = llvm.getelementptr %[[V795]]{{\[}}%[[V793]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V797:.+]] = llvm.ptrtoint %[[V796]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V798:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V799:.+]] = llvm.add %[[V797]], %[[V798]] : i64
// CHECK-NEXT:    %[[V800:.+]] = llvm.call @malloc(%[[V799]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V801:.+]] = llvm.ptrtoint %[[V800]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V802:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V803:.+]] = llvm.sub %[[V798]], %[[V802]] : i64
// CHECK-NEXT:    %[[V804:.+]] = llvm.add %[[V801]], %[[V803]] : i64
// CHECK-NEXT:    %[[V805:.+]] = llvm.urem %[[V804]], %[[V798]] : i64
// CHECK-NEXT:    %[[V806:.+]] = llvm.sub %[[V804]], %[[V805]] : i64
// CHECK-NEXT:    %[[V807:.+]] = llvm.inttoptr %[[V806]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V808:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V809:.+]] = llvm.insertvalue %[[V800]], %[[V808]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V810:.+]] = llvm.insertvalue %[[V807]], %[[V809]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V811:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V812:.+]] = llvm.insertvalue %[[V811]], %[[V810]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V813:.+]] = llvm.insertvalue %[[V793]], %[[V812]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V814:.+]] = llvm.insertvalue %[[V794]], %[[V813]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V815:.+]] = llvm.extractvalue %[[V814]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V816:.+]] = llvm.getelementptr inbounds|nuw %[[V815]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V816]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V817:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V818:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V819:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V820:.+]] = llvm.getelementptr %[[V819]]{{\[}}%[[V817]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V821:.+]] = llvm.ptrtoint %[[V820]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V822:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V823:.+]] = llvm.add %[[V821]], %[[V822]] : i64
// CHECK-NEXT:    %[[V824:.+]] = llvm.call @malloc(%[[V823]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V825:.+]] = llvm.ptrtoint %[[V824]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V826:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V827:.+]] = llvm.sub %[[V822]], %[[V826]] : i64
// CHECK-NEXT:    %[[V828:.+]] = llvm.add %[[V825]], %[[V827]] : i64
// CHECK-NEXT:    %[[V829:.+]] = llvm.urem %[[V828]], %[[V822]] : i64
// CHECK-NEXT:    %[[V830:.+]] = llvm.sub %[[V828]], %[[V829]] : i64
// CHECK-NEXT:    %[[V831:.+]] = llvm.inttoptr %[[V830]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V832:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V833:.+]] = llvm.insertvalue %[[V824]], %[[V832]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V834:.+]] = llvm.insertvalue %[[V831]], %[[V833]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V835:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V836:.+]] = llvm.insertvalue %[[V835]], %[[V834]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V837:.+]] = llvm.insertvalue %[[V817]], %[[V836]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V838:.+]] = llvm.insertvalue %[[V818]], %[[V837]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V839:.+]] = llvm.extractvalue %[[V838]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V840:.+]] = llvm.getelementptr inbounds|nuw %[[V839]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V724]], %[[V840]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V841:.+]] = llvm.extractvalue %[[V838]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V842:.+]] = llvm.getelementptr inbounds|nuw %[[V841]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V727]], %[[V842]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V843:.+]] = llvm.extractvalue %[[V838]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V844:.+]] = llvm.getelementptr inbounds|nuw %[[V843]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V730]], %[[V844]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V845:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V846:.+]] = llvm.getelementptr inbounds|nuw %[[V845]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V847:.+]] = llvm.load %[[V846]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V848:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V849:.+]] = llvm.getelementptr inbounds|nuw %[[V848]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V850:.+]] = llvm.load %[[V849]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V851:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V852:.+]] = llvm.getelementptr inbounds|nuw %[[V851]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V853:.+]] = llvm.load %[[V852]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V854:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V855:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V856:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V857:.+]] = llvm.getelementptr %[[V856]]{{\[}}%[[V854]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V858:.+]] = llvm.ptrtoint %[[V857]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V859:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V860:.+]] = llvm.add %[[V858]], %[[V859]] : i64
// CHECK-NEXT:    %[[V861:.+]] = llvm.call @malloc(%[[V860]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V862:.+]] = llvm.ptrtoint %[[V861]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V863:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V864:.+]] = llvm.sub %[[V859]], %[[V863]] : i64
// CHECK-NEXT:    %[[V865:.+]] = llvm.add %[[V862]], %[[V864]] : i64
// CHECK-NEXT:    %[[V866:.+]] = llvm.urem %[[V865]], %[[V859]] : i64
// CHECK-NEXT:    %[[V867:.+]] = llvm.sub %[[V865]], %[[V866]] : i64
// CHECK-NEXT:    %[[V868:.+]] = llvm.inttoptr %[[V867]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V869:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V870:.+]] = llvm.insertvalue %[[V861]], %[[V869]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V871:.+]] = llvm.insertvalue %[[V868]], %[[V870]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V872:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V873:.+]] = llvm.insertvalue %[[V872]], %[[V871]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V874:.+]] = llvm.insertvalue %[[V854]], %[[V873]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V875:.+]] = llvm.insertvalue %[[V855]], %[[V874]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V876:.+]] = llvm.extractvalue %[[V875]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V877:.+]] = llvm.getelementptr inbounds|nuw %[[V876]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V847]], %[[V877]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V878:.+]] = llvm.extractvalue %[[V875]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V879:.+]] = llvm.getelementptr inbounds|nuw %[[V878]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V850]], %[[V879]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V880:.+]] = llvm.extractvalue %[[V875]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V881:.+]] = llvm.getelementptr inbounds|nuw %[[V880]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V853]], %[[V881]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V882:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V883:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V884:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V885:.+]] = llvm.getelementptr %[[V884]]{{\[}}%[[V882]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V886:.+]] = llvm.ptrtoint %[[V885]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V887:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V888:.+]] = llvm.add %[[V886]], %[[V887]] : i64
// CHECK-NEXT:    %[[V889:.+]] = llvm.call @malloc(%[[V888]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V890:.+]] = llvm.ptrtoint %[[V889]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V891:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V892:.+]] = llvm.sub %[[V887]], %[[V891]] : i64
// CHECK-NEXT:    %[[V893:.+]] = llvm.add %[[V890]], %[[V892]] : i64
// CHECK-NEXT:    %[[V894:.+]] = llvm.urem %[[V893]], %[[V887]] : i64
// CHECK-NEXT:    %[[V895:.+]] = llvm.sub %[[V893]], %[[V894]] : i64
// CHECK-NEXT:    %[[V896:.+]] = llvm.inttoptr %[[V895]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V897:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V898:.+]] = llvm.insertvalue %[[V889]], %[[V897]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V899:.+]] = llvm.insertvalue %[[V896]], %[[V898]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V900:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V901:.+]] = llvm.insertvalue %[[V900]], %[[V899]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V902:.+]] = llvm.insertvalue %[[V882]], %[[V901]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V903:.+]] = llvm.insertvalue %[[V883]], %[[V902]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.br ^bb22(%[[V48]] : i64)
// CHECK-NEXT:    ^bb22(%[[V904:.+]]: i64):
// CHECK-NEXT:    %[[V905:.+]] = llvm.icmp "slt" %[[V904]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V905]], ^bb23, ^bb28
// CHECK-NEXT:    ^bb23:
// CHECK-NEXT:    %[[V906:.+]] = llvm.icmp "ult" %[[V904]], %[[V48]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V906]], ^bb24, ^bb25
// CHECK-NEXT:    ^bb24:
// CHECK-NEXT:    llvm.br ^bb26(%[[V47]] : i64)
// CHECK-NEXT:    ^bb25:
// CHECK-NEXT:    %[[V907:.+]] = llvm.extractvalue %[[V838]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V908:.+]] = llvm.getelementptr inbounds|nuw %[[V907]]{{\[}}%[[V904]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V909:.+]] = llvm.load %[[V908]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V910:.+]] = llvm.extractvalue %[[V875]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V911:.+]] = llvm.getelementptr inbounds|nuw %[[V910]]{{\[}}%[[V904]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V912:.+]] = llvm.load %[[V911]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V913:.+]] = llvm.icmp "eq" %[[V912]], %[[V47]] : i64
// CHECK-NEXT:    %[[V914:.+]] = llvm.select %[[V913]], %[[V909]], %[[V912]] : i1, i64
// CHECK-NEXT:    llvm.br ^bb26(%[[V914]] : i64)
// CHECK-NEXT:    ^bb26(%[[V915:.+]]: i64):
// CHECK-NEXT:    llvm.br ^bb27
// CHECK-NEXT:    ^bb27:
// CHECK-NEXT:    %[[V916:.+]] = llvm.extractvalue %[[V903]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V917:.+]] = llvm.getelementptr inbounds|nuw %[[V916]]{{\[}}%[[V904]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V915]], %[[V917]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V918:.+]] = llvm.add %[[V904]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb22(%[[V918]] : i64)
// CHECK-NEXT:    ^bb28:
// CHECK-NEXT:    %[[V919:.+]] = llvm.extractvalue %[[V875]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V919]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V920:.+]] = llvm.extractvalue %[[V838]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V920]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.br ^bb29(%[[V48]], %[[V47]] : i64, i64)
// CHECK-NEXT:    ^bb29(%[[V921:.+]]: i64, %[[V922:.+]]: i64):
// CHECK-NEXT:    %[[V923:.+]] = llvm.icmp "slt" %[[V921]], %[[V49]] : i64
// CHECK-NEXT:    llvm.cond_br %[[V923]], ^bb30, ^bb31
// CHECK-NEXT:    ^bb30:
// CHECK-NEXT:    %[[V924:.+]] = llvm.extractvalue %[[V903]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V925:.+]] = llvm.getelementptr inbounds|nuw %[[V924]]{{\[}}%[[V921]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V926:.+]] = llvm.load %[[V925]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V927:.+]] = llvm.mul %[[V926]], %[[V922]] : i64
// CHECK-NEXT:    %[[V928:.+]] = llvm.add %[[V921]], %[[V47]] : i64
// CHECK-NEXT:    llvm.br ^bb29(%[[V928]], %[[V927]] : i64, i64)
// CHECK-NEXT:    ^bb31:
// CHECK-NEXT:    %[[V929:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V930:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V931:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V932:.+]] = llvm.getelementptr %[[V931]]{{\[}}%[[V929]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V933:.+]] = llvm.ptrtoint %[[V932]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V934:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V935:.+]] = llvm.add %[[V933]], %[[V934]] : i64
// CHECK-NEXT:    %[[V936:.+]] = llvm.call @malloc(%[[V935]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V937:.+]] = llvm.ptrtoint %[[V936]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V938:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V939:.+]] = llvm.sub %[[V934]], %[[V938]] : i64
// CHECK-NEXT:    %[[V940:.+]] = llvm.add %[[V937]], %[[V939]] : i64
// CHECK-NEXT:    %[[V941:.+]] = llvm.urem %[[V940]], %[[V934]] : i64
// CHECK-NEXT:    %[[V942:.+]] = llvm.sub %[[V940]], %[[V941]] : i64
// CHECK-NEXT:    %[[V943:.+]] = llvm.inttoptr %[[V942]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V944:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V945:.+]] = llvm.insertvalue %[[V936]], %[[V944]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V946:.+]] = llvm.insertvalue %[[V943]], %[[V945]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V947:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V948:.+]] = llvm.insertvalue %[[V947]], %[[V946]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V949:.+]] = llvm.insertvalue %[[V929]], %[[V948]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V950:.+]] = llvm.insertvalue %[[V930]], %[[V949]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V951:.+]] = llvm.extractvalue %[[V950]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V952:.+]] = llvm.getelementptr inbounds|nuw %[[V951]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V952]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V953:.+]] = llvm.extractvalue %[[V950]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V954:.+]] = llvm.getelementptr inbounds|nuw %[[V953]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V922]], %[[V954]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V955:.+]] = llvm.extractvalue %[[V903]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V956:.+]] = llvm.getelementptr inbounds|nuw %[[V955]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V957:.+]] = llvm.load %[[V956]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V958:.+]] = llvm.extractvalue %[[V903]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V959:.+]] = llvm.getelementptr inbounds|nuw %[[V958]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V960:.+]] = llvm.load %[[V959]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V961:.+]] = llvm.extractvalue %[[V903]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V962:.+]] = llvm.getelementptr inbounds|nuw %[[V961]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V963:.+]] = llvm.load %[[V962]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V964:.+]] = llvm.extractvalue %[[V950]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V965:.+]] = llvm.getelementptr inbounds|nuw %[[V964]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V966:.+]] = llvm.load %[[V965]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V967:.+]] = llvm.mul %[[V957]], %[[V960]] : i64
// CHECK-NEXT:    %[[V968:.+]] = llvm.mul %[[V967]], %[[V963]] : i64
// CHECK-NEXT:    %[[V969:.+]] = llvm.add %[[V968]], %[[V20]] : i64
// CHECK-NEXT:    %[[V970:.+]] = llvm.udiv %[[V969]], %[[V21]] : i64
// CHECK-NEXT:    %[[V971:.+]] = llvm.mul %[[V970]], %[[V21]] : i64
// CHECK-NEXT:    %[[V972:.+]] = llvm.mul %[[V966]], %[[V17]] : i64
// CHECK-NEXT:    %[[V973:.+]] = llvm.add %[[V972]], %[[V20]] : i64
// CHECK-NEXT:    %[[V974:.+]] = llvm.udiv %[[V973]], %[[V21]] : i64
// CHECK-NEXT:    %[[V975:.+]] = llvm.mul %[[V974]], %[[V21]] : i64
// CHECK-NEXT:    %[[V976:.+]] = llvm.add %[[V971]], %[[V975]] : i64
// CHECK-NEXT:    %[[V977:.+]] = llvm.add %[[V976]], %[[V21]] : i64
// CHECK-NEXT:    %[[V978:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V979:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V978]], %[[V977]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V980:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V981:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V982:.+]] = llvm.insertvalue %[[V979]], %[[V981]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V983:.+]] = llvm.insertvalue %[[V979]], %[[V982]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V984:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V985:.+]] = llvm.insertvalue %[[V984]], %[[V983]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V986:.+]] = llvm.insertvalue %[[V977]], %[[V985]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V987:.+]] = llvm.insertvalue %[[V980]], %[[V986]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V988:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V989:.+]] = llvm.extractvalue %[[V987]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V990:.+]] = llvm.insertvalue %[[V989]], %[[V988]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V991:.+]] = llvm.extractvalue %[[V987]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V992:.+]] = llvm.getelementptr %[[V991]]{{\[}}%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V993:.+]] = llvm.insertvalue %[[V992]], %[[V990]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V994:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V995:.+]] = llvm.insertvalue %[[V994]], %[[V993]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V996:.+]] = llvm.insertvalue %[[V963]], %[[V995]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V997:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V998:.+]] = llvm.insertvalue %[[V997]], %[[V996]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V999:.+]] = llvm.insertvalue %[[V960]], %[[V998]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1000:.+]] = llvm.mul %[[V997]], %[[V963]] : i64
// CHECK-NEXT:    %[[V1001:.+]] = llvm.insertvalue %[[V1000]], %[[V999]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1002:.+]] = llvm.insertvalue %[[V957]], %[[V1001]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1003:.+]] = llvm.mul %[[V1000]], %[[V960]] : i64
// CHECK-NEXT:    %[[V1004:.+]] = llvm.insertvalue %[[V1003]], %[[V1002]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1005:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1006:.+]] = llvm.extractvalue %[[V987]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1007:.+]] = llvm.insertvalue %[[V1006]], %[[V1005]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1008:.+]] = llvm.extractvalue %[[V987]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1009:.+]] = llvm.getelementptr %[[V1008]]{{\[}}%[[V971]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1010:.+]] = llvm.insertvalue %[[V1009]], %[[V1007]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1011:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1012:.+]] = llvm.insertvalue %[[V1011]], %[[V1010]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1013:.+]] = llvm.insertvalue %[[V966]], %[[V1012]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1014:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1015:.+]] = llvm.insertvalue %[[V1014]], %[[V1013]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1016:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1017:.+]] = llvm.insertvalue %[[V1016]], %[[V1015]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1018:.+]] = llvm.mul %[[V1014]], %[[V966]] : i64
// CHECK-NEXT:    %[[V1019:.+]] = llvm.insertvalue %[[V1018]], %[[V1017]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1020:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1021:.+]] = llvm.extractvalue %[[V987]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1022:.+]] = llvm.insertvalue %[[V1021]], %[[V1020]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1023:.+]] = llvm.extractvalue %[[V987]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1024:.+]] = llvm.getelementptr %[[V1023]]{{\[}}%[[V976]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1025:.+]] = llvm.insertvalue %[[V1024]], %[[V1022]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1026:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1027:.+]] = llvm.insertvalue %[[V1026]], %[[V1025]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1028:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1029:.+]] = llvm.insertvalue %[[V1028]], %[[V1027]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1030:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1031:.+]] = llvm.insertvalue %[[V1030]], %[[V1029]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1032:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1033:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1034:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1035:.+]] = llvm.getelementptr %[[V1034]]{{\[}}%[[V1032]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1036:.+]] = llvm.ptrtoint %[[V1035]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1037:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1038:.+]] = llvm.add %[[V1036]], %[[V1037]] : i64
// CHECK-NEXT:    %[[V1039:.+]] = llvm.call @malloc(%[[V1038]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1040:.+]] = llvm.ptrtoint %[[V1039]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1041:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1042:.+]] = llvm.sub %[[V1037]], %[[V1041]] : i64
// CHECK-NEXT:    %[[V1043:.+]] = llvm.add %[[V1040]], %[[V1042]] : i64
// CHECK-NEXT:    %[[V1044:.+]] = llvm.urem %[[V1043]], %[[V1037]] : i64
// CHECK-NEXT:    %[[V1045:.+]] = llvm.sub %[[V1043]], %[[V1044]] : i64
// CHECK-NEXT:    %[[V1046:.+]] = llvm.inttoptr %[[V1045]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1047:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1048:.+]] = llvm.insertvalue %[[V1039]], %[[V1047]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1049:.+]] = llvm.insertvalue %[[V1046]], %[[V1048]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1050:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1051:.+]] = llvm.insertvalue %[[V1050]], %[[V1049]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1052:.+]] = llvm.insertvalue %[[V1032]], %[[V1051]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1053:.+]] = llvm.insertvalue %[[V1033]], %[[V1052]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1054:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1055:.+]] = llvm.extractvalue %[[V516]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1056:.+]] = llvm.alloca %[[V1054]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1057:.+]] = llvm.extractvalue %[[V762]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1058:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1059:.+]] = llvm.getelementptr %[[V1056]]{{\[}}%[[V1058]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1057]], %[[V1059]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1060:.+]] = llvm.extractvalue %[[V762]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1061:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1062:.+]] = llvm.getelementptr %[[V1056]]{{\[}}%[[V1061]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1060]], %[[V1062]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1063:.+]] = llvm.extractvalue %[[V762]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1064:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V1065:.+]] = llvm.getelementptr %[[V1056]]{{\[}}%[[V1064]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1063]], %[[V1065]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1066:.+]] = llvm.alloca %[[V1054]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1067:.+]] = llvm.extractvalue %[[V1004]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1068:.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:    %[[V1069:.+]] = llvm.getelementptr %[[V1066]]{{\[}}%[[V1068]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1067]], %[[V1069]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1070:.+]] = llvm.extractvalue %[[V1004]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1071:.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT:    %[[V1072:.+]] = llvm.getelementptr %[[V1066]]{{\[}}%[[V1071]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1070]], %[[V1072]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1073:.+]] = llvm.extractvalue %[[V1004]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1074:.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT:    %[[V1075:.+]] = llvm.getelementptr %[[V1066]]{{\[}}%[[V1074]]] : (!llvm.ptr, i32) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1073]], %[[V1075]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1076:.+]] = llvm.extractvalue %[[V762]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1077:.+]] = llvm.extractvalue %[[V1004]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1078:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1079:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1080:.+]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V1081:.+]] = llvm.call @wrap_expand(%[[ARG0]], %[[V1076]], %[[V1055]], %[[V1077]], %[[V1056]], %[[V1078]], %[[V1066]], %[[V1079]], %[[V1080]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1082:.+]] = llvm.extractvalue %[[V516]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1082]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1083:.+]] = llvm.extractvalue %[[V1004]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1084:.+]] = llvm.extractvalue %[[V1004]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1085:.+]] = llvm.extractvalue %[[V1004]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1086:.+]] = llvm.mul %[[V1083]], %[[V1084]] : i64
// CHECK-NEXT:    %[[V1087:.+]] = llvm.mul %[[V1086]], %[[V1085]] : i64
// CHECK-NEXT:    %[[V1088:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1089:.+]] = llvm.extractvalue %[[V1019]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1090:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1091:.+]] = llvm.alloca %[[V1090]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1092:.+]] = llvm.getelementptr %[[V1091]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1083]], %[[V1092]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1093:.+]] = llvm.getelementptr %[[V1091]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1084]], %[[V1093]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1094:.+]] = llvm.getelementptr %[[V1091]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1085]], %[[V1094]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1095:.+]] = llvm.extractvalue %[[V1004]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1096:.+]] = llvm.extractvalue %[[V1019]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1097:.+]] = llvm.extractvalue %[[V1031]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1098:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1099:.+]] = llvm.mlir.constant(7 : i64) : i64
// CHECK-NEXT:    %[[V1100:.+]] = llvm.call @wrap_nonzero(%[[ARG0]], %[[V1095]], %[[V1096]], %[[V1097]], %[[V1087]], %[[V1098]], %[[V1091]], %[[V1089]], %[[V1099]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, i64, i64, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1101:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1102:.+]] = llvm.getelementptr %[[V1101]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1103:.+]] = llvm.ptrtoint %[[V1102]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1104:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1105:.+]] = llvm.mul %[[V1103]], %[[V1104]] : i64
// CHECK-NEXT:    %[[V1106:.+]] = llvm.extractvalue %[[V1053]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1107:.+]] = llvm.extractvalue %[[V1031]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1108:.+]] = llvm.call @wrap_copy_d2h(%[[ARG0]], %[[V1106]], %[[V1107]], %[[V1105]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr<1>, i64) -> i32
// CHECK-NEXT:    %[[V1109:.+]] = llvm.extractvalue %[[V903]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1109]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1110:.+]] = llvm.extractvalue %[[V950]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1110]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1111:.+]] = llvm.extractvalue %[[V814]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1111]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1112:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V1113:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1114:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1115:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1116:.+]] = llvm.getelementptr %[[V1115]]{{\[}}%[[V1113]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1117:.+]] = llvm.ptrtoint %[[V1116]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1118:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1119:.+]] = llvm.add %[[V1117]], %[[V1118]] : i64
// CHECK-NEXT:    %[[V1120:.+]] = llvm.call @malloc(%[[V1119]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1121:.+]] = llvm.ptrtoint %[[V1120]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1122:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1123:.+]] = llvm.sub %[[V1118]], %[[V1122]] : i64
// CHECK-NEXT:    %[[V1124:.+]] = llvm.add %[[V1121]], %[[V1123]] : i64
// CHECK-NEXT:    %[[V1125:.+]] = llvm.urem %[[V1124]], %[[V1118]] : i64
// CHECK-NEXT:    %[[V1126:.+]] = llvm.sub %[[V1124]], %[[V1125]] : i64
// CHECK-NEXT:    %[[V1127:.+]] = llvm.inttoptr %[[V1126]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1128:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1129:.+]] = llvm.insertvalue %[[V1120]], %[[V1128]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1130:.+]] = llvm.insertvalue %[[V1127]], %[[V1129]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1131:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1132:.+]] = llvm.insertvalue %[[V1131]], %[[V1130]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1133:.+]] = llvm.insertvalue %[[V1113]], %[[V1132]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1134:.+]] = llvm.insertvalue %[[V1114]], %[[V1133]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1135:.+]] = llvm.extractvalue %[[V1134]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1136:.+]] = llvm.getelementptr inbounds|nuw %[[V1135]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V47]], %[[V1136]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1137:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1138:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1139:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1140:.+]] = llvm.getelementptr %[[V1139]]{{\[}}%[[V1137]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1141:.+]] = llvm.ptrtoint %[[V1140]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1142:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1143:.+]] = llvm.add %[[V1141]], %[[V1142]] : i64
// CHECK-NEXT:    %[[V1144:.+]] = llvm.call @malloc(%[[V1143]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1145:.+]] = llvm.ptrtoint %[[V1144]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1146:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1147:.+]] = llvm.sub %[[V1142]], %[[V1146]] : i64
// CHECK-NEXT:    %[[V1148:.+]] = llvm.add %[[V1145]], %[[V1147]] : i64
// CHECK-NEXT:    %[[V1149:.+]] = llvm.urem %[[V1148]], %[[V1142]] : i64
// CHECK-NEXT:    %[[V1150:.+]] = llvm.sub %[[V1148]], %[[V1149]] : i64
// CHECK-NEXT:    %[[V1151:.+]] = llvm.inttoptr %[[V1150]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1152:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1153:.+]] = llvm.insertvalue %[[V1144]], %[[V1152]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1154:.+]] = llvm.insertvalue %[[V1151]], %[[V1153]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1155:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1156:.+]] = llvm.insertvalue %[[V1155]], %[[V1154]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1157:.+]] = llvm.insertvalue %[[V1137]], %[[V1156]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1158:.+]] = llvm.insertvalue %[[V1138]], %[[V1157]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1159:.+]] = llvm.extractvalue %[[V1158]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1160:.+]] = llvm.getelementptr inbounds|nuw %[[V1159]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V23]], %[[V1160]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1161:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1162:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1163:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1164:.+]] = llvm.getelementptr %[[V1163]]{{\[}}%[[V1161]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1165:.+]] = llvm.ptrtoint %[[V1164]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1166:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1167:.+]] = llvm.add %[[V1165]], %[[V1166]] : i64
// CHECK-NEXT:    %[[V1168:.+]] = llvm.call @malloc(%[[V1167]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1169:.+]] = llvm.ptrtoint %[[V1168]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1170:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1171:.+]] = llvm.sub %[[V1166]], %[[V1170]] : i64
// CHECK-NEXT:    %[[V1172:.+]] = llvm.add %[[V1169]], %[[V1171]] : i64
// CHECK-NEXT:    %[[V1173:.+]] = llvm.urem %[[V1172]], %[[V1166]] : i64
// CHECK-NEXT:    %[[V1174:.+]] = llvm.sub %[[V1172]], %[[V1173]] : i64
// CHECK-NEXT:    %[[V1175:.+]] = llvm.inttoptr %[[V1174]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1176:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1177:.+]] = llvm.insertvalue %[[V1168]], %[[V1176]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1178:.+]] = llvm.insertvalue %[[V1175]], %[[V1177]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1179:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1180:.+]] = llvm.insertvalue %[[V1179]], %[[V1178]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1181:.+]] = llvm.insertvalue %[[V1161]], %[[V1180]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1182:.+]] = llvm.insertvalue %[[V1162]], %[[V1181]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1183:.+]] = llvm.extractvalue %[[V1053]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1184:.+]] = llvm.getelementptr inbounds|nuw %[[V1183]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:    %[[V1185:.+]] = llvm.load %[[V1184]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[V1186:.+]] = llvm.extractvalue %[[V1053]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1186]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1187:.+]] = llvm.sext %[[V1185]] : i32 to i64
// CHECK-NEXT:    %[[V1188:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1189:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1190:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1191:.+]] = llvm.getelementptr %[[V1190]]{{\[}}%[[V1188]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1192:.+]] = llvm.ptrtoint %[[V1191]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1193:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1194:.+]] = llvm.add %[[V1192]], %[[V1193]] : i64
// CHECK-NEXT:    %[[V1195:.+]] = llvm.call @malloc(%[[V1194]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1196:.+]] = llvm.ptrtoint %[[V1195]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1197:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1198:.+]] = llvm.sub %[[V1193]], %[[V1197]] : i64
// CHECK-NEXT:    %[[V1199:.+]] = llvm.add %[[V1196]], %[[V1198]] : i64
// CHECK-NEXT:    %[[V1200:.+]] = llvm.urem %[[V1199]], %[[V1193]] : i64
// CHECK-NEXT:    %[[V1201:.+]] = llvm.sub %[[V1199]], %[[V1200]] : i64
// CHECK-NEXT:    %[[V1202:.+]] = llvm.inttoptr %[[V1201]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1203:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1204:.+]] = llvm.insertvalue %[[V1195]], %[[V1203]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1205:.+]] = llvm.insertvalue %[[V1202]], %[[V1204]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1206:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1207:.+]] = llvm.insertvalue %[[V1206]], %[[V1205]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1208:.+]] = llvm.insertvalue %[[V1188]], %[[V1207]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1209:.+]] = llvm.insertvalue %[[V1189]], %[[V1208]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1210:.+]] = llvm.extractvalue %[[V1209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1211:.+]] = llvm.getelementptr inbounds|nuw %[[V1210]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V49]], %[[V1211]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1212:.+]] = llvm.extractvalue %[[V1209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1213:.+]] = llvm.getelementptr inbounds|nuw %[[V1212]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1187]], %[[V1213]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1214:.+]] = llvm.extractvalue %[[V1209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1215:.+]] = llvm.getelementptr inbounds|nuw %[[V1214]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1216:.+]] = llvm.load %[[V1215]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1217:.+]] = llvm.extractvalue %[[V1209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1218:.+]] = llvm.getelementptr inbounds|nuw %[[V1217]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1219:.+]] = llvm.load %[[V1218]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1220:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1221:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1222:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1223:.+]] = llvm.getelementptr %[[V1222]]{{\[}}%[[V1220]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1224:.+]] = llvm.ptrtoint %[[V1223]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1225:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1226:.+]] = llvm.add %[[V1224]], %[[V1225]] : i64
// CHECK-NEXT:    %[[V1227:.+]] = llvm.call @malloc(%[[V1226]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1228:.+]] = llvm.ptrtoint %[[V1227]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1229:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1230:.+]] = llvm.sub %[[V1225]], %[[V1229]] : i64
// CHECK-NEXT:    %[[V1231:.+]] = llvm.add %[[V1228]], %[[V1230]] : i64
// CHECK-NEXT:    %[[V1232:.+]] = llvm.urem %[[V1231]], %[[V1225]] : i64
// CHECK-NEXT:    %[[V1233:.+]] = llvm.sub %[[V1231]], %[[V1232]] : i64
// CHECK-NEXT:    %[[V1234:.+]] = llvm.inttoptr %[[V1233]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1235:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1236:.+]] = llvm.insertvalue %[[V1227]], %[[V1235]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1237:.+]] = llvm.insertvalue %[[V1234]], %[[V1236]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1238:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1239:.+]] = llvm.insertvalue %[[V1238]], %[[V1237]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1240:.+]] = llvm.insertvalue %[[V1220]], %[[V1239]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1241:.+]] = llvm.insertvalue %[[V1221]], %[[V1240]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1242:.+]] = llvm.extractvalue %[[V1241]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1243:.+]] = llvm.getelementptr inbounds|nuw %[[V1242]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1216]], %[[V1243]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1244:.+]] = llvm.extractvalue %[[V1241]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1245:.+]] = llvm.getelementptr inbounds|nuw %[[V1244]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1219]], %[[V1245]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1246:.+]] = llvm.extractvalue %[[V1209]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1247:.+]] = llvm.getelementptr inbounds|nuw %[[V1246]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1248:.+]] = llvm.load %[[V1247]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1249:.+]] = llvm.extractvalue %[[V1241]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1250:.+]] = llvm.getelementptr inbounds|nuw %[[V1249]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1251:.+]] = llvm.load %[[V1250]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1252:.+]] = llvm.mul %[[V1248]], %[[V17]] : i64
// CHECK-NEXT:    %[[V1253:.+]] = llvm.mul %[[V1251]], %[[V17]] : i64
// CHECK-NEXT:    %[[V1254:.+]] = llvm.intr.umax(%[[V1252]], %[[V1253]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1255:.+]] = llvm.add %[[V1254]], %[[V20]] : i64
// CHECK-NEXT:    %[[V1256:.+]] = llvm.udiv %[[V1255]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1257:.+]] = llvm.mul %[[V1256]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1258:.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT:    %[[V1259:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V1258]], %[[V1257]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V1260:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1261:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1262:.+]] = llvm.insertvalue %[[V1259]], %[[V1261]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1263:.+]] = llvm.insertvalue %[[V1259]], %[[V1262]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1264:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1265:.+]] = llvm.insertvalue %[[V1264]], %[[V1263]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1266:.+]] = llvm.insertvalue %[[V1257]], %[[V1265]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1267:.+]] = llvm.insertvalue %[[V1260]], %[[V1266]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1268:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1269:.+]] = llvm.extractvalue %[[V1267]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1270:.+]] = llvm.insertvalue %[[V1269]], %[[V1268]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1271:.+]] = llvm.extractvalue %[[V1267]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1272:.+]] = llvm.getelementptr %[[V1271]]{{\[}}%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1273:.+]] = llvm.insertvalue %[[V1272]], %[[V1270]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1274:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1275:.+]] = llvm.insertvalue %[[V1274]], %[[V1273]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1276:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1277:.+]] = llvm.insertvalue %[[V1276]], %[[V1275]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1278:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1279:.+]] = llvm.insertvalue %[[V1278]], %[[V1277]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1280:.+]] = llvm.insertvalue %[[V1251]], %[[V1279]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1281:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1282:.+]] = llvm.insertvalue %[[V1281]], %[[V1280]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1283:.+]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT:    %[[V1284:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1285:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1286:.+]] = llvm.getelementptr %[[V1285]]{{\[}}%[[V1283]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1287:.+]] = llvm.ptrtoint %[[V1286]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1288:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1289:.+]] = llvm.add %[[V1287]], %[[V1288]] : i64
// CHECK-NEXT:    %[[V1290:.+]] = llvm.call @malloc(%[[V1289]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1291:.+]] = llvm.ptrtoint %[[V1290]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1292:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1293:.+]] = llvm.sub %[[V1288]], %[[V1292]] : i64
// CHECK-NEXT:    %[[V1294:.+]] = llvm.add %[[V1291]], %[[V1293]] : i64
// CHECK-NEXT:    %[[V1295:.+]] = llvm.urem %[[V1294]], %[[V1288]] : i64
// CHECK-NEXT:    %[[V1296:.+]] = llvm.sub %[[V1294]], %[[V1295]] : i64
// CHECK-NEXT:    %[[V1297:.+]] = llvm.inttoptr %[[V1296]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1298:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1299:.+]] = llvm.insertvalue %[[V1290]], %[[V1298]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1300:.+]] = llvm.insertvalue %[[V1297]], %[[V1299]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1301:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1302:.+]] = llvm.insertvalue %[[V1301]], %[[V1300]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1303:.+]] = llvm.insertvalue %[[V1283]], %[[V1302]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1304:.+]] = llvm.insertvalue %[[V1284]], %[[V1303]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1305:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1306:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1307:.+]] = llvm.getelementptr %[[V1306]]{{\[}}%[[V1305]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1308:.+]] = llvm.ptrtoint %[[V1307]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1309:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1310:.+]] = llvm.add %[[V1308]], %[[V1309]] : i64
// CHECK-NEXT:    %[[V1311:.+]] = llvm.call @malloc(%[[V1310]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1312:.+]] = llvm.ptrtoint %[[V1311]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1313:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1314:.+]] = llvm.sub %[[V1309]], %[[V1313]] : i64
// CHECK-NEXT:    %[[V1315:.+]] = llvm.add %[[V1312]], %[[V1314]] : i64
// CHECK-NEXT:    %[[V1316:.+]] = llvm.urem %[[V1315]], %[[V1309]] : i64
// CHECK-NEXT:    %[[V1317:.+]] = llvm.sub %[[V1315]], %[[V1316]] : i64
// CHECK-NEXT:    %[[V1318:.+]] = llvm.inttoptr %[[V1317]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1319:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1320:.+]] = llvm.insertvalue %[[V1311]], %[[V1319]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1321:.+]] = llvm.insertvalue %[[V1318]], %[[V1320]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1322:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1323:.+]] = llvm.insertvalue %[[V1322]], %[[V1321]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1324:.+]] = llvm.extractvalue %[[V1019]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1325:.+]] = llvm.extractvalue %[[V1019]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1326:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1327:.+]] = llvm.insertvalue %[[V1324]], %[[V1326]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1328:.+]] = llvm.insertvalue %[[V1325]], %[[V1327]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1329:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1330:.+]] = llvm.insertvalue %[[V1329]], %[[V1328]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1331:.+]] = llvm.extractvalue %[[V1019]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1332:.+]] = llvm.extractvalue %[[V1019]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1333:.+]] = llvm.extractvalue %[[V1019]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1334:.+]] = llvm.extractvalue %[[V1019]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1335:.+]] = llvm.extractvalue %[[V1019]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1336:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1337:.+]] = llvm.extractvalue %[[V1330]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1338:.+]] = llvm.extractvalue %[[V1330]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
// CHECK-NEXT:    %[[V1339:.+]] = llvm.insertvalue %[[V1337]], %[[V1336]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1340:.+]] = llvm.insertvalue %[[V1338]], %[[V1339]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1341:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1342:.+]] = llvm.insertvalue %[[V1341]], %[[V1340]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1343:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1344:.+]] = llvm.insertvalue %[[V1343]], %[[V1342]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1345:.+]] = llvm.insertvalue %[[V1334]], %[[V1344]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1346:.+]] = llvm.insertvalue %[[V1248]], %[[V1345]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1347:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1348:.+]] = llvm.insertvalue %[[V1347]], %[[V1346]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1349:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1350:.+]] = llvm.extractvalue %[[V1348]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1351:.+]] = llvm.mul %[[V1349]], %[[V1350]] : i64
// CHECK-NEXT:    %[[V1352:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1353:.+]] = llvm.alloca %[[V1352]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1354:.+]] = llvm.getelementptr %[[V1353]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1349]], %[[V1354]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1355:.+]] = llvm.getelementptr %[[V1353]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1350]], %[[V1355]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1356:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1357:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1358:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1359:.+]] = llvm.alloca %[[V1358]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1360:.+]] = llvm.getelementptr %[[V1359]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1356]], %[[V1360]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1361:.+]] = llvm.getelementptr %[[V1359]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1357]], %[[V1361]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1362:.+]] = llvm.extractvalue %[[V1348]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1363:.+]] = llvm.extractvalue %[[V1282]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1364:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1365:.+]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-NEXT:    %[[V1366:.+]] = llvm.call @wrap_transpose(%[[ARG0]], %[[V1362]], %[[V1363]], %[[V1364]], %[[V1353]], %[[V1359]], %[[V1351]], %[[V1365]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, i64, !llvm.ptr, !llvm.ptr, i64, i64) -> i32
// CHECK-NEXT:    %[[V1367:.+]] = llvm.extractvalue %[[V1304]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1368:.+]] = llvm.getelementptr inbounds|nuw %[[V1367]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1251]], %[[V1368]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1369:.+]] = llvm.extractvalue %[[V1304]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1370:.+]] = llvm.getelementptr inbounds|nuw %[[V1369]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V16]], %[[V1370]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1371:.+]] = llvm.extractvalue %[[V1304]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1372:.+]] = llvm.getelementptr inbounds|nuw %[[V1371]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1373:.+]] = llvm.load %[[V1372]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1374:.+]] = llvm.extractvalue %[[V1323]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    llvm.store %[[V1373]], %[[V1374]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1375:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1376:.+]] = llvm.extractvalue %[[V1323]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1377:.+]] = llvm.extractvalue %[[V1323]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    %[[V1378:.+]] = llvm.insertvalue %[[V1376]], %[[V1375]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1379:.+]] = llvm.insertvalue %[[V1377]], %[[V1378]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1380:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1381:.+]] = llvm.insertvalue %[[V1380]], %[[V1379]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1382:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1383:.+]] = llvm.insertvalue %[[V1382]], %[[V1381]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1384:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1385:.+]] = llvm.insertvalue %[[V1384]], %[[V1383]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1386:.+]] = llvm.extractvalue %[[V1209]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1386]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1387:.+]] = llvm.extractvalue %[[V1241]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1387]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1388:.+]] = llvm.extractvalue %[[V1304]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1388]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1389:.+]] = llvm.extractvalue %[[V1158]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1389]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1390:.+]] = llvm.extractvalue %[[V1182]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1390]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1391:.+]] = llvm.extractvalue %[[V1134]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1391]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1392:.+]] = llvm.call @hipdnn_ep_stream_sync(%[[ARG0]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    %[[V1393:.+]] = llvm.extractvalue %[[V540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1394:.+]] = llvm.extractvalue %[[V1385]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1395:.+]] = llvm.getelementptr inbounds|nuw %[[V1394]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1396:.+]] = llvm.load %[[V1395]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1397:.+]] = llvm.icmp "slt" %[[V1396]], %[[V48]] : i64
// CHECK-NEXT:    %[[V1398:.+]] = llvm.add %[[V1396]], %[[V1393]] : i64
// CHECK-NEXT:    %[[V1399:.+]] = llvm.select %[[V1397]], %[[V1398]], %[[V1396]] : i1, i64
// CHECK-NEXT:    %[[V1400:.+]] = llvm.intr.smin(%[[V1393]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1401:.+]] = llvm.intr.smax(%[[V1399]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1402:.+]] = llvm.intr.smin(%[[V1401]], %[[V1393]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1403:.+]] = llvm.sub %[[V1402]], %[[V1400]] : i64
// CHECK-NEXT:    %[[V1404:.+]] = llvm.intr.smax(%[[V1403]], %[[V48]]) : (i64, i64) -> i64
// CHECK-NEXT:    %[[V1405:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1406:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1407:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1408:.+]] = llvm.getelementptr %[[V1407]]{{\[}}%[[V1405]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1409:.+]] = llvm.ptrtoint %[[V1408]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1410:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1411:.+]] = llvm.add %[[V1409]], %[[V1410]] : i64
// CHECK-NEXT:    %[[V1412:.+]] = llvm.call @malloc(%[[V1411]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1413:.+]] = llvm.ptrtoint %[[V1412]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1414:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1415:.+]] = llvm.sub %[[V1410]], %[[V1414]] : i64
// CHECK-NEXT:    %[[V1416:.+]] = llvm.add %[[V1413]], %[[V1415]] : i64
// CHECK-NEXT:    %[[V1417:.+]] = llvm.urem %[[V1416]], %[[V1410]] : i64
// CHECK-NEXT:    %[[V1418:.+]] = llvm.sub %[[V1416]], %[[V1417]] : i64
// CHECK-NEXT:    %[[V1419:.+]] = llvm.inttoptr %[[V1418]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1420:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1421:.+]] = llvm.insertvalue %[[V1412]], %[[V1420]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1422:.+]] = llvm.insertvalue %[[V1419]], %[[V1421]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1423:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1424:.+]] = llvm.insertvalue %[[V1423]], %[[V1422]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1425:.+]] = llvm.insertvalue %[[V1405]], %[[V1424]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1426:.+]] = llvm.insertvalue %[[V1406]], %[[V1425]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1427:.+]] = llvm.extractvalue %[[V1426]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1428:.+]] = llvm.getelementptr inbounds|nuw %[[V1427]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1404]], %[[V1428]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1429:.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:    %[[V1430:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1431:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1432:.+]] = llvm.getelementptr %[[V1431]]{{\[}}%[[V1429]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1433:.+]] = llvm.ptrtoint %[[V1432]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1434:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-NEXT:    %[[V1435:.+]] = llvm.add %[[V1433]], %[[V1434]] : i64
// CHECK-NEXT:    %[[V1436:.+]] = llvm.call @malloc(%[[V1435]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1437:.+]] = llvm.ptrtoint %[[V1436]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1438:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1439:.+]] = llvm.sub %[[V1434]], %[[V1438]] : i64
// CHECK-NEXT:    %[[V1440:.+]] = llvm.add %[[V1437]], %[[V1439]] : i64
// CHECK-NEXT:    %[[V1441:.+]] = llvm.urem %[[V1440]], %[[V1434]] : i64
// CHECK-NEXT:    %[[V1442:.+]] = llvm.sub %[[V1440]], %[[V1441]] : i64
// CHECK-NEXT:    %[[V1443:.+]] = llvm.inttoptr %[[V1442]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[V1444:.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1445:.+]] = llvm.insertvalue %[[V1436]], %[[V1444]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1446:.+]] = llvm.insertvalue %[[V1443]], %[[V1445]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1447:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1448:.+]] = llvm.insertvalue %[[V1447]], %[[V1446]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1449:.+]] = llvm.insertvalue %[[V1429]], %[[V1448]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1450:.+]] = llvm.insertvalue %[[V1430]], %[[V1449]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1451:.+]] = llvm.extractvalue %[[V1450]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1452:.+]] = llvm.getelementptr inbounds|nuw %[[V1451]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V439]], %[[V1452]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1453:.+]] = llvm.extractvalue %[[V1450]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1454:.+]] = llvm.getelementptr inbounds|nuw %[[V1453]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V442]], %[[V1454]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1455:.+]] = llvm.extractvalue %[[V1450]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1456:.+]] = llvm.getelementptr inbounds|nuw %[[V1455]]{{\[}}%[[V23]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V24]], %[[V1456]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1457:.+]] = llvm.extractvalue %[[V1426]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1458:.+]] = llvm.getelementptr inbounds|nuw %[[V1457]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1459:.+]] = llvm.load %[[V1458]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1460:.+]] = llvm.mul %[[V1459]], %[[V23]] : i64
// CHECK-NEXT:    %[[V1461:.+]] = llvm.add %[[V1460]], %[[V20]] : i64
// CHECK-NEXT:    %[[V1462:.+]] = llvm.udiv %[[V1461]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1463:.+]] = llvm.mul %[[V1462]], %[[V21]] : i64
// CHECK-NEXT:    %[[V1464:.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT:    %[[V1465:.+]] = llvm.call @hipdnn_ep_get_pool_base(%[[ARG0]], %[[V1464]], %[[V1463]]) : (!llvm.ptr, i32, i64) -> !llvm.ptr<1>
// CHECK-NEXT:    %[[V1466:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1467:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1468:.+]] = llvm.insertvalue %[[V1465]], %[[V1467]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1469:.+]] = llvm.insertvalue %[[V1465]], %[[V1468]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1470:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1471:.+]] = llvm.insertvalue %[[V1470]], %[[V1469]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1472:.+]] = llvm.insertvalue %[[V1463]], %[[V1471]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1473:.+]] = llvm.insertvalue %[[V1466]], %[[V1472]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1474:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1475:.+]] = llvm.extractvalue %[[V1473]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1476:.+]] = llvm.insertvalue %[[V1475]], %[[V1474]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1477:.+]] = llvm.extractvalue %[[V1473]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1478:.+]] = llvm.getelementptr %[[V1477]]{{\[}}%[[V48]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
// CHECK-NEXT:    %[[V1479:.+]] = llvm.insertvalue %[[V1478]], %[[V1476]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1480:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1481:.+]] = llvm.insertvalue %[[V1480]], %[[V1479]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1482:.+]] = llvm.insertvalue %[[V1459]], %[[V1481]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1483:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1484:.+]] = llvm.insertvalue %[[V1483]], %[[V1482]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1485:.+]] = llvm.extractvalue %[[V1450]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1486:.+]] = llvm.getelementptr inbounds|nuw %[[V1485]]{{\[}}%[[V48]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1487:.+]] = llvm.load %[[V1486]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1488:.+]] = llvm.extractvalue %[[V1450]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1489:.+]] = llvm.getelementptr inbounds|nuw %[[V1488]]{{\[}}%[[V47]]] : (!llvm.ptr, i64) -> !llvm.ptr, i64
// CHECK-NEXT:    %[[V1490:.+]] = llvm.load %[[V1489]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[V1491:.+]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-NEXT:    %[[V1492:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:    %[[V1493:.+]] = llvm.mul %[[V1491]], %[[V1490]] : i64
// CHECK-NEXT:    %[[V1494:.+]] = llvm.mul %[[V1493]], %[[V1487]] : i64
// CHECK-NEXT:    %[[V1495:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:    %[[V1496:.+]] = llvm.getelementptr %[[V1495]]{{\[}}%[[V1494]]] : (!llvm.ptr, i64) -> !llvm.ptr, f16
// CHECK-NEXT:    %[[V1497:.+]] = llvm.ptrtoint %[[V1496]] : !llvm.ptr to i64
// CHECK-NEXT:    %[[V1498:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1499:.+]] = llvm.alloca %[[V1498]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1500:.+]] = llvm.getelementptr %[[V1499]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1487]], %[[V1500]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1501:.+]] = llvm.getelementptr %[[V1499]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1490]], %[[V1501]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1502:.+]] = llvm.getelementptr %[[V1499]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1491]], %[[V1502]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1503:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1504:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1505:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1506:.+]] = llvm.call @hipdnn_ep_alloc_output(%[[ARG0]], %[[V1503]], %[[V1499]], %[[V1504]], %[[V1505]]) : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1507:.+]] = llvm.addrspacecast %[[V1506]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT:    %[[V1508:.+]] = llvm.mlir.poison : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1509:.+]] = llvm.insertvalue %[[V1507]], %[[V1508]][0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1510:.+]] = llvm.insertvalue %[[V1507]], %[[V1509]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1511:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:    %[[V1512:.+]] = llvm.insertvalue %[[V1511]], %[[V1510]][2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1513:.+]] = llvm.insertvalue %[[V1487]], %[[V1512]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1514:.+]] = llvm.insertvalue %[[V1490]], %[[V1513]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1515:.+]] = llvm.insertvalue %[[V1491]], %[[V1514]][3, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1516:.+]] = llvm.insertvalue %[[V1493]], %[[V1515]][4, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1517:.+]] = llvm.insertvalue %[[V1491]], %[[V1516]][4, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1518:.+]] = llvm.insertvalue %[[V1492]], %[[V1517]][4, 2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1519:.+]] = llvm.extractvalue %[[V540]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1520:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1521:.+]] = llvm.alloca %[[V1520]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1522:.+]] = llvm.getelementptr %[[V1521]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1519]], %[[V1522]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1523:.+]] = llvm.extractvalue %[[V1484]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1524:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1525:.+]] = llvm.alloca %[[V1524]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1526:.+]] = llvm.getelementptr %[[V1525]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1523]], %[[V1526]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1527:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1528:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1529:.+]] = llvm.alloca %[[V1528]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1530:.+]] = llvm.getelementptr %[[V1529]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1527]], %[[V1530]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1531:.+]] = llvm.extractvalue %[[V1385]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1532:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1533:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1534:.+]] = llvm.alloca %[[V1533]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1535:.+]] = llvm.getelementptr %[[V1534]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1532]], %[[V1535]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1536:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1537:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1538:.+]] = llvm.alloca %[[V1537]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1539:.+]] = llvm.getelementptr %[[V1538]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1536]], %[[V1539]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1540:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1541:.+]] = llvm.extractvalue %[[V540]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1542:.+]] = llvm.extractvalue %[[V1484]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1543:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1544:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1545:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1546:.+]] = llvm.call @wrap_slice(%[[ARG0]], %[[V1541]], %[[V1529]], %[[V1531]], %[[V1534]], %[[V1538]], %[[V1542]], %[[V1521]], %[[V1543]], %[[V1525]], %[[V1544]], %[[V1540]], %[[V1540]], %[[V1540]], %[[V1545]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V1547:.+]] = llvm.extractvalue %[[V1323]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK-NEXT:    llvm.call @free(%[[V1547]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1548:.+]] = llvm.extractvalue %[[V494]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1549:.+]] = llvm.extractvalue %[[V494]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1550:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V1551:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1552:.+]] = llvm.alloca %[[V1551]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1553:.+]] = llvm.getelementptr %[[V1552]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1548]], %[[V1553]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1554:.+]] = llvm.getelementptr %[[V1552]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1549]], %[[V1554]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1555:.+]] = llvm.getelementptr %[[V1552]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1550]], %[[V1555]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1556:.+]] = llvm.extractvalue %[[V1282]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1557:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1558:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1559:.+]] = llvm.alloca %[[V1558]] x !llvm.array<2 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1560:.+]] = llvm.getelementptr %[[V1559]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1556]], %[[V1560]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1561:.+]] = llvm.getelementptr %[[V1559]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1557]], %[[V1561]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1562:.+]] = llvm.extractvalue %[[V1484]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1563:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1564:.+]] = llvm.alloca %[[V1563]] x !llvm.array<1 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1565:.+]] = llvm.getelementptr %[[V1564]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1562]], %[[V1565]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1566:.+]] = llvm.extractvalue %[[V1518]][3, 0] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1567:.+]] = llvm.extractvalue %[[V1518]][3, 1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1568:.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-NEXT:    %[[V1569:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1570:.+]] = llvm.alloca %[[V1569]] x !llvm.array<3 x i64> {alignment = 8 : i64} : (i64) -> !llvm.ptr
// CHECK-NEXT:    %[[V1571:.+]] = llvm.getelementptr %[[V1570]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1566]], %[[V1571]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1572:.+]] = llvm.getelementptr %[[V1570]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1567]], %[[V1572]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1573:.+]] = llvm.getelementptr %[[V1570]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:    llvm.store %[[V1568]], %[[V1573]] : i64, !llvm.ptr
// CHECK-NEXT:    %[[V1574:.+]] = llvm.extractvalue %[[V494]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1575:.+]] = llvm.extractvalue %[[V1282]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[V1576:.+]] = llvm.extractvalue %[[V1484]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    %[[V1577:.+]] = llvm.extractvalue %[[V1518]][1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    %[[V1578:.+]] = llvm.mlir.zero : !llvm.ptr<1>
// CHECK-NEXT:    %[[V1579:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1580:.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT:    %[[V1581:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1582:.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[V1583:.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %[[V1584:.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NEXT:    %[[V1585:.+]] = llvm.call @wrap_scatter_nd(%[[ARG0]], %[[V1574]], %[[V1575]], %[[V1576]], %[[V1577]], %[[V1578]], %[[V1552]], %[[V1579]], %[[V1559]], %[[V1580]], %[[V1564]], %[[V1581]], %[[V1570]], %[[V1582]], %[[V1583]], %[[V1584]]) : (!llvm.ptr, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i64, i64, i64) -> i32
// CHECK-NEXT:    %[[V1586:.+]] = llvm.extractvalue %[[V1426]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1586]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    %[[V1587:.+]] = llvm.extractvalue %[[V1450]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:    llvm.call @free(%[[V1587]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:    llvm.return %[[V1518]] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT:    }
// CHECK-LABEL: llvm.func @inference_init(
// CHECK-SAME:    %[[ARG0:[^,]*]]: !llvm.ptr, %[[ARG1:[^,]*]]: !llvm.ptr, %[[ARG2:[^,]*]]: !llvm.ptr) -> i32 attributes {llvm.emit_c_interface, sym_visibility = "public"} {
// CHECK-NEXT:    %[[V0:.+]] = llvm.mlir.addressof @__metadata_blob : !llvm.ptr
// CHECK-NEXT:    %[[V1:.+]] = llvm.mlir.constant(312 : i64) : i64
// CHECK-NEXT:    %[[V2:.+]] = llvm.call @hipdnn_ep_state_init_with_fs(%[[ARG0]], %[[ARG1]], %[[V0]], %[[V1]], %[[ARG2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
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

module {
  func.func @main_graph(%arg0: tensor<?x?xi64> {onnx.name = "input_ids"}, %arg1: tensor<?x4096xf16> {onnx.name = "image_features"}) -> (tensor<?x?x4096xf16> {onnx.name = "inputs_embeds"}) attributes {onnx.graph.name = "main_graph"} {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Constant"() {node.outputs = ["embed_tokens.weight"], location = "embedding.onnx.data", offset = 0 : i64, size = 2034237440 : i64} : () -> tensor<248320x4096xf16>
    %2 = "onnx.Constant"() {node.outputs = ["/Constant_output_0"], value = dense<248056> : tensor<i64>} : () -> tensor<i64>
    %3 = "onnx.Constant"() {node.outputs = ["/Constant_1_output_0"], value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
    %4 = "onnx.Constant"() {node.outputs = ["/Constant_3_output_0"], value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %5 = "onnx.Constant"() {node.outputs = ["/Constant_4_output_0"], value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
    %6 = "onnx.Reshape"(%arg1, %3) {allowzero = 0 : si64, node.outputs = ["/Reshape_output_0"], onnx_node_name = "/Reshape"} : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
    %7 = "onnx.Equal"(%arg0, %2) {node.outputs = ["/Equal_output_0"], onnx_node_name = "/Equal"} : (tensor<?x?xi64>, tensor<i64>) -> tensor<?x?xi1>
    %8 = "onnx.Unsqueeze"(%7, %3) {node.outputs = ["/Unsqueeze_output_0"], onnx_node_name = "/Unsqueeze"} : (tensor<?x?xi1>, tensor<1xi64>) -> tensor<?x?x1xi1>
    %9 = "onnx.Gather"(%1, %arg0) {axis = 0 : si64, node.outputs = ["/embed_tokens/Gather_output_0"], onnx_node_name = "/embed_tokens/Gather"} : (tensor<248320x4096xf16>, tensor<?x?xi64>) -> tensor<?x?x4096xf16>
    %10 = "onnx.Shape"(%9) {node.outputs = ["/Shape_1_output_0"], onnx_node_name = "/Shape_1", start = 0 : si64} : (tensor<?x?x4096xf16>) -> tensor<3xi64>
    %11 = "onnx.Expand"(%8, %10) {node.outputs = ["/Expand_output_0"], onnx_node_name = "/Expand"} : (tensor<?x?x1xi1>, tensor<3xi64>) -> tensor<?x?x?xi1>
    %12 = "onnx.Expand"(%11, %10) {node.outputs = ["/Expand_1_output_0"], onnx_node_name = "/Expand_1"} : (tensor<?x?x?xi1>, tensor<3xi64>) -> tensor<?x?x?xi1>
    %13 = "onnx.NonZero"(%12) {node.outputs = ["/NonZero_output_0"], onnx_node_name = "/NonZero"} : (tensor<?x?x?xi1>) -> tensor<3x?xi64>
    %14 = "onnx.Transpose"(%13) {node.outputs = ["/Transpose_output_0"], onnx_node_name = "/Transpose", perm = [1, 0]} : (tensor<3x?xi64>) -> tensor<?x3xi64>
    %15 = "onnx.Shape"(%14) {node.outputs = ["/Shape_2_output_0"], onnx_node_name = "/Shape_2", start = 0 : si64} : (tensor<?x3xi64>) -> tensor<2xi64>
    %16 = "onnx.Gather"(%15, %4) {axis = 0 : si64, node.outputs = ["/Gather_output_0"], onnx_node_name = "/Gather"} : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    %17 = "onnx.Unsqueeze"(%16, %5) {node.outputs = ["/Unsqueeze_1_output_0"], onnx_node_name = "/Unsqueeze_1"} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
    %18 = "onnx.Slice"(%6, %5, %17, %5, %0) {node.outputs = ["/Slice_output_0"], onnx_node_name = "/Slice"} : (tensor<?xf16>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, none) -> tensor<?xf16>
    %19 = "onnx.ScatterND"(%9, %14, %18) {node.outputs = ["inputs_embeds"], onnx_node_name = "/ScatterND", reduction = "none"} : (tensor<?x?x4096xf16>, tensor<?x3xi64>, tensor<?xf16>) -> tensor<?x?x4096xf16>
    "onnx.Return"(%19) : (tensor<?x?x4096xf16>) -> ()
  }
}
