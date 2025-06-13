# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from builtin.io import _printf
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm
from gpu.id import thread_idx
from gpu.mma import mma
from gpu.mma_util import store_matrix_d, load_matrix_a, load_matrix_b
from sys import _RegisterPackType, is_nvidia_gpu, llvm_intrinsic, sizeof
from memory import UnsafePointer, bitcast

# See https://github.com/llvm/llvm-project/blob/fe28ea37b640ea4842583df3b89e08877220fb8e/mlir/test/Target/LLVMIR/Import/nvvmir.ll
# and https://github.com/llvm/llvm-project/blob/fe28ea37b640ea4842583df3b89e08877220fb8e/mlir/test/Target/LLVMIR/nvvmir.mlir

fn mma_sync_16x8x16_bf16_fp32():
    a = SIMD[DType.bfloat16, 8](1.0)
    b = SIMD[DType.bfloat16, 4](1.0)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )

fn mma_with_load[M: Int, N:Int, K:Int](A: UnsafePointer[BFloat16], B: UnsafePointer[BFloat16]):
    a = load_matrix_a[M, N, K](A, 0, 0, K)
    b = load_matrix_b[M, N, K](B, 0, 0, N)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : a: %g, %g, %g, %g %g, %g, %g, %g, b: %g, %g, %g, %g, d: %g %g %g %g\n"](
        thread_idx.x,
        a[0].cast[DType.float64](),
        a[1].cast[DType.float64](),
        a[2].cast[DType.float64](),
        a[3].cast[DType.float64](),
        a[4].cast[DType.float64](),
        a[5].cast[DType.float64](),
        a[6].cast[DType.float64](),
        a[7].cast[DType.float64](),
        b[0].cast[DType.float64](),
        b[1].cast[DType.float64](),
        b[2].cast[DType.float64](),
        b[3].cast[DType.float64](),
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )

fn mma_with_load_and_store[M: Int, N:Int, K:Int](A: UnsafePointer[BFloat16], B: UnsafePointer[BFloat16], D:UnsafePointer[Float32]):
    a = load_matrix_a[M, N, K](A, 0, 0, K)
    b = load_matrix_b[M, N, K](B, 0, 0, N)
    c = SIMD[DType.float32, 4](0.0)
    d = SIMD[DType.float32, 4](0.0)
    mma(d, a, b, c)

    store_matrix_d[M, N, K](
        D,
        d,
        0,
        0,
        N
    )


fn mma_wrapper(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    var sa = a.split()
    var ssa = sa[0].split()
    var ssa1 = sa[1].split()
    var sb = b.split()
    var sc = c.split()

    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k16.row.col.f16.f16",
        _RegisterPackType[SIMD[DType.float16, 2], SIMD[DType.float16, 2]],
    ](ssa[0], ssa0[1], ssa1[0], ssa1[1], sb[0], sb[1], sc[0], sc[1])

    d = rebind[__type_of(d)](r[0].join(r[1]))

fn mma_sync_16x8x16_fp16_fp16():
    a = SIMD[DType.float16, 8](1.0)
    b = SIMD[DType.float16, 4](1.0)
    c = SIMD[DType.float16, 4](0.0)
    d = SIMD[DType.float16, 4](0.0)
    mma_wrapper(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[DType.float64](),
        d[1].cast[DType.float64](),
        d[2].cast[DType.float64](),
        d[3].cast[DType.float64](),
    )

fn mma_with_store[M: Int, N:Int, K:Int](D: UnsafePointer[Float16]):
    
    a = SIMD[DType.float16, 8](1.0)
    b = SIMD[DType.float16, 4](1.0)
    c = SIMD[DType.float16, 4](0.0)
    d = SIMD[DType.float16, 4](0.0)
    mma_wrapper(d, a, b, c)

    store_matrix_d[M, N, K](
        D,
        d,
        0,
        0,
        N,
    )

def print_compiled_mma_sync_16x8x16_bf16_fp32():
    print("== mma_sync_16x8x16_bf16_fp32 compiled")

    print(_compile_code_asm[mma_sync_16x8x16_bf16_fp32, emission_kind="llvm"]())

def print_compiled_mma_sync_16x8x16_fp16_fp16():
    print("== mma_sync_16x8x16_fp16_fp16 compiled")

    print(_compile_code_asm[mma_sync_16x8x16_fp16_fp16, emission_kind="llvm"]())


def test_mma_sync_16x8x16_bf16_fp32(ctx: DeviceContext):
    print("== mma_sync_16x8x16_bf16_fp32")
    ctx.enqueue_function[mma_sync_16x8x16_bf16_fp32](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()

def test_mma_sync_16x8x16_fp16_fp16(ctx: DeviceContext):
    print("== mma_sync_16x8x16_fp16_fp16")
    ctx.enqueue_function[mma_sync_16x8x16_fp16_fp16](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()

def test_mma_with_store(ctx: DeviceContext):
    print("== mma_with_store")
    alias M = 16
    alias N = 8
    alias K = 16

    D = ctx.enqueue_create_buffer[DType.float16](M * N).enqueue_fill(0)
    D_ptr = D.unsafe_ptr()

    ctx.enqueue_function[mma_with_store[M, N, K]](
        D_ptr,
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()

    with D.map_to_host() as D_host:
        for i in range(M):
            for j in range(N):
                idx = i * N + j
                print(D_host[idx], end= ";")
            print()

def test_mma_with_load(ctx: DeviceContext):
    print("== mma_with_load")
    alias M = 16
    alias N = 8
    alias K = 16

    A = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    with A.map_to_host() as A_host:
        for i in range(M):
            for j in range(K):
                A_host[i * K + j] = i * K + j
                print(A_host[i * K + j], end=";")
            print()
    A_ptr = A.unsafe_ptr()

    B = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    with B.map_to_host() as B_host:
        print("B matrix")
        for i in range(K):
            for j in range(N):
                B_host[i * N + j] = i * N + j
                print(B_host[i * N + j], end=";")
            print()
    B_ptr = B.unsafe_ptr()

    ctx.enqueue_function[mma_with_load[M, N, K]](
        A_ptr,
        B_ptr,
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()

def test_mma_with_load_and_store(ctx: DeviceContext):
    print("== mma_with_load")
    alias M = 16
    alias N = 8
    alias K = 16

    A = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    with A.map_to_host() as A_host:
        for i in range(M):
            for j in range(K):
                A_host[i * K + j] = i * K + j
    A_ptr = A.unsafe_ptr()
    B = ctx.enqueue_create_buffer[DType.bfloat16](K * N)
    with B.map_to_host() as B_host:
        for i in range(K):
            for j in range(N):
                B_host[i * N + j] = i * N + j
    B_ptr = B.unsafe_ptr()

    D = ctx.enqueue_create_buffer[DType.float32](M * N).enqueue_fill(0)
    D_ptr = D.unsafe_ptr()

    ctx.enqueue_function[mma_with_load_and_store[M, N, K]](
        A_ptr,
        B_ptr,
        D_ptr,
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()

    with D.map_to_host() as D_host:
        for i in range(M):
            for j in range(N):
                idx = i * N + j
                print(D_host[idx], end= ";")
            print()

def main():
    #print_compiled_mma_sync_16x8x16_bf16_fp32()
    #print()
    #print_compiled_mma_sync_16x8x16_fp16_fp16()
    with DeviceContext() as ctx:
        #test_mma_sync_16x8x16_bf16_fp32(ctx)
        #test_mma_sync_16x8x16_fp16_fp16(ctx)
        #test_mma_with_store(ctx)
        test_mma_with_load_and_store(ctx)
