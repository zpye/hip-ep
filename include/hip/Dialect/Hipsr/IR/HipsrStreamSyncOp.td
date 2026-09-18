//
// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//

//===----------------------------------------------------------------------===//
// Hipsr_StreamSyncOp
//===----------------------------------------------------------------------===//

def Hipsr_StreamSyncOp : Hipsr_Op<"stream_sync",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Wait until the context's compute stream is idle";
  let description = [{
    Waits until every op already queued on the context's stream has finished.

    The host may then read values those ops wrote, including host buffers
    filled by `hipsr.copy_d2h`.

    Example:
    ```mlir
    %0 = hipsr.pool_domain(%ctx, %input) { ... }
    hipsr.stream_sync(%ctx)
    %1 = hipsr.pool_domain(%ctx, %0) { ... }
    ```
  }];

  let arguments = (ins Hipsr_ContextType:$ctx);
  let results = (outs);
  let assemblyFormat = "`(` $ctx `)` attr-dict `:` type($ctx)";
}
