# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused SwiGLU (Gate + Up + SiLU Epilogue + Sub-Channel Quantization) Pallas TPU Kernel.

Supports both:
  1. 3-way decoupled software pipeline (pipeline_mode='pipelined'):
     - Stationary LHS loaded once per M-token chunk.
     - Decoupled MXU GEMM on tile j overlapping with VPU SwiGLU + FP8 Quant on
     tile j-1
       and DMA prefetch/store.
  2. Standard compiler-scheduled grid (pipeline_mode='grid').
"""

import functools
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def swiglu_kernel_grid(
    lhs_ref,
    lhs_scale_ref,
    w_gate_ref,
    w_gate_scale_ref,
    w_up_ref,
    w_up_scale_ref,
    out_q_ref,
    out_scale_ref=None,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    quant_out: bool = False,
    quant_mode: str = "subchannel",
    subchannel_k: int = 512,
    quant_max: float = 448.0,
):
  """Standard grid-based SwiGLU kernel body with in-register FP8 quantization."""
  del tile_k
  # 1. Compute Gate and Up MXU tiles in float32 accumulators
  acc_gate = jnp.dot(
      lhs_ref[...], w_gate_ref[...], preferred_element_type=jnp.float32
  )
  acc_up = jnp.dot(
      lhs_ref[...], w_up_ref[...], preferred_element_type=jnp.float32
  )

  # 2. Dequantize in VPU registers
  gate_scaled = acc_gate.astype(jnp.bfloat16) * (
      lhs_scale_ref[...] * w_gate_scale_ref[...]
  )
  up_scaled = acc_up.astype(jnp.bfloat16) * (
      lhs_scale_ref[...] * w_up_scale_ref[...]
  )

  # 3. In-register SwiGLU activation
  res = jax.nn.silu(gate_scaled) * up_scaled

  if not quant_out:
    out_q_ref[...] = res.astype(out_q_ref.dtype)
  else:
    if quant_mode == "channelwise":
      abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12)
      res_scaled = res / scale
      res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
      out_q_ref[...] = res_clamped.astype(jnp.float8_e4m3fn)
      out_scale_ref[...] = jnp.reshape(scale, (-1,)).astype(jnp.bfloat16)
    elif subchannel_k <= 0 or subchannel_k >= tile_n:
      abs_max = jnp.max(jnp.abs(res), axis=-1, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12)
      res_scaled = res / scale
      res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
      out_q_ref[...] = res_clamped.astype(jnp.float8_e4m3fn)
      out_scale_ref[...] = jnp.reshape(scale, (-1,)).astype(jnp.bfloat16)
    else:
      num_blocks = tile_n // subchannel_k
      res_blocked = jnp.reshape(res, (tile_m, num_blocks, subchannel_k))
      abs_max = jnp.max(jnp.abs(res_blocked), axis=-1, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12)
      res_scaled = res_blocked / scale
      res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
      out_q_ref[...] = jnp.reshape(res_clamped, (tile_m, tile_n)).astype(
          jnp.float8_e4m3fn
      )
      out_scale_ref[...] = jnp.reshape(scale, (-1,)).astype(jnp.bfloat16)


def _pipelined_swiglu_kernel(
    *args,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_n_tiles: int,
    quant_out: bool = False,
    quant_mode: str = "subchannel",
    subchannel_k: int = 512,
    quant_max: float = 448.0,
):
  """3-Way Decoupled Software-Pipelined SwiGLU Kernel Body.

  Stationary LHS in VMEM; inner N-loop with concurrent MXU (GEMM Tile j),
  VPU (SwiGLU + Quant Tile j-1), DMA Ingress (W[j+1]), and DMA Egress
  (Out[j-2]).
  """
  idx = 0
  lhs_ref = args[idx]
  idx += 1
  lhs_scale_ref = args[idx]
  idx += 1
  w_gate_hbm_ref = args[idx]
  idx += 1
  w_gate_scale_hbm_ref = args[idx]
  idx += 1
  w_up_hbm_ref = args[idx]
  idx += 1
  w_up_scale_hbm_ref = args[idx]
  idx += 1

  if not quant_out:
    out_hbm_ref = args[idx]
    idx += 1
    out_scale_hbm_ref = None
  else:
    out_q_hbm_ref = args[idx]
    idx += 1
    out_scale_hbm_ref = args[idx]
    idx += 1

  sems = args[idx:]
  sem_wg = sems[0:2]
  sem_wgs = sems[2:4]
  sem_wu = sems[4:6]
  sem_wus = sems[6:8]
  sem_out = sems[8:10]
  if quant_out:
    sem_out_scale = sems[10:12]

  # Stationary LHS loaded once per M-tile
  m_idx = pl.program_id(0)
  m_start = m_idx * tile_m
  m_slice = pl.ds(m_start, tile_m)
  lhs_tile = lhs_ref[...]
  lhs_scale_tile = lhs_scale_ref[...]

  w_scale_rows = w_gate_scale_hbm_ref.shape[0]
  if quant_mode == "channelwise":
    tile_scale_elements = tile_n
  else:
    num_sub_k = (
        1
        if (subchannel_k <= 0 or subchannel_k >= tile_n)
        else (tile_n // subchannel_k)
    )
    tile_scale_elements = tile_m * num_sub_k

  scoped_vmem = dict(
      w_gate_vmem=pltpu.VMEM((2, tile_k, tile_n), w_gate_hbm_ref.dtype),
      w_gate_scale_vmem=pltpu.VMEM(
          (2, w_scale_rows, tile_n), w_gate_scale_hbm_ref.dtype
      ),
      w_up_vmem=pltpu.VMEM((2, tile_k, tile_n), w_up_hbm_ref.dtype),
      w_up_scale_vmem=pltpu.VMEM(
          (2, w_scale_rows, tile_n), w_up_scale_hbm_ref.dtype
      ),
      acc_gate_vmem=pltpu.VMEM((2, tile_m, tile_n), jnp.float32),
      acc_up_vmem=pltpu.VMEM((2, tile_m, tile_n), jnp.float32),
  )
  if not quant_out:
    scoped_vmem["out_vmem"] = pltpu.VMEM((2, tile_m, tile_n), out_hbm_ref.dtype)
  else:
    scoped_vmem["out_q_vmem"] = pltpu.VMEM(
        (2, tile_m, tile_n), out_q_hbm_ref.dtype
    )
    scoped_vmem["out_scale_vmem_0"] = pltpu.VMEM(
        (tile_scale_elements,), out_scale_hbm_ref.dtype
    )
    scoped_vmem["out_scale_vmem_1"] = pltpu.VMEM(
        (tile_scale_elements,), out_scale_hbm_ref.dtype
    )

  @functools.partial(pl.run_scoped, **scoped_vmem)
  def _run_pipeline(
      w_gate_vmem,
      w_gate_scale_vmem,
      w_up_vmem,
      w_up_scale_vmem,
      acc_gate_vmem,
      acc_up_vmem,
      out_vmem=None,
      out_q_vmem=None,
      out_scale_vmem_0=None,
      out_scale_vmem_1=None,
  ):
    out_scale_vmems = (
        (out_scale_vmem_0, out_scale_vmem_1) if quant_out else None
    )

    def load_weights(j, slot):
      n_start = j * tile_n
      n_slice = pl.ds(n_start, tile_n)
      k_slice = pl.ds(0, tile_k)
      scale_k_slice = pl.ds(0, w_scale_rows)

      with jax.named_scope("hbm_load_weights"):
        copy_wg = pltpu.make_async_copy(
            src_ref=w_gate_hbm_ref.at[k_slice, n_slice],
            dst_ref=w_gate_vmem.at[slot, :, :],
            sem=sem_wg[slot],
        )
        copy_wg.start()

        copy_wgs = pltpu.make_async_copy(
            src_ref=w_gate_scale_hbm_ref.at[scale_k_slice, n_slice],
            dst_ref=w_gate_scale_vmem.at[slot, :, :],
            sem=sem_wgs[slot],
        )
        copy_wgs.start()

        copy_wu = pltpu.make_async_copy(
            src_ref=w_up_hbm_ref.at[k_slice, n_slice],
            dst_ref=w_up_vmem.at[slot, :, :],
            sem=sem_wu[slot],
        )
        copy_wu.start()

        copy_wus = pltpu.make_async_copy(
            src_ref=w_up_scale_hbm_ref.at[scale_k_slice, n_slice],
            dst_ref=w_up_scale_vmem.at[slot, :, :],
            sem=sem_wus[slot],
        )
        copy_wus.start()

      return copy_wg, copy_wgs, copy_wu, copy_wus

    def wait_weights(descs):
      with jax.named_scope("hbm_wait_weights"):
        descs[0].wait()
        descs[1].wait()
        descs[2].wait()
        descs[3].wait()

    def dispatch_store(j, out_slot):
      n_start = j * tile_n
      n_slice = pl.ds(n_start, tile_n)

      with jax.named_scope("hbm_store_output"):
        if not quant_out:
          copy_out = pltpu.make_async_copy(
              src_ref=out_vmem.at[out_slot, :, :],
              dst_ref=out_hbm_ref.at[m_slice, n_slice],
              sem=sem_out[out_slot],
          )
          copy_out.start()
          return (copy_out,)
        else:
          copy_q = pltpu.make_async_copy(
              src_ref=out_q_vmem.at[out_slot, :, :],
              dst_ref=out_q_hbm_ref.at[m_slice, n_slice],
              sem=sem_out[out_slot],
          )
          copy_q.start()

          scale_offset = (m_idx * num_n_tiles + j) * tile_scale_elements
          vmem_scale = out_scale_vmems[out_slot]
          copy_s = pltpu.make_async_copy(
              src_ref=vmem_scale,
              dst_ref=out_scale_hbm_ref.at[
                  pl.ds(scale_offset, tile_scale_elements)
              ],
              sem=sem_out_scale[out_slot],
          )
          copy_s.start()
          return copy_q, copy_s

    def wait_store(store_descs):
      if store_descs is not None:
        with jax.named_scope("hbm_wait_store"):
          for desc in store_descs:
            desc.wait()

    def compute_vpu_step(j_prev, acc_slot, out_slot):
      del j_prev
      with jax.named_scope("vpu_swiglu_and_quant"):
        g_s = acc_gate_vmem[acc_slot, :, :].astype(jnp.bfloat16) * (
            lhs_scale_tile * w_gate_scale_vmem[acc_slot, :, :]
        )
        u_s = acc_up_vmem[acc_slot, :, :].astype(jnp.bfloat16) * (
            lhs_scale_tile * w_up_scale_vmem[acc_slot, :, :]
        )
        res = jax.nn.silu(g_s) * u_s

        if not quant_out:
          out_vmem[out_slot, :, :] = res.astype(out_hbm_ref.dtype)
        else:
          vmem_scale = out_scale_vmems[out_slot]
          if quant_mode == "channelwise":
            abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)
            scale = jnp.maximum(abs_max / quant_max, 1e-12)
            res_scaled = res / scale
            res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
            out_q_vmem[out_slot, :, :] = res_clamped.astype(jnp.float8_e4m3fn)
            vmem_scale[...] = jnp.squeeze(scale, axis=0).astype(jnp.bfloat16)
          elif subchannel_k <= 0 or subchannel_k >= tile_n:
            abs_max = jnp.max(jnp.abs(res), axis=-1, keepdims=True)
            scale = jnp.maximum(abs_max / quant_max, 1e-12)
            res_scaled = res / scale
            res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
            out_q_vmem[out_slot, :, :] = res_clamped.astype(jnp.float8_e4m3fn)
            vmem_scale[...] = jnp.squeeze(scale, axis=-1).astype(jnp.bfloat16)
          else:
            num_blocks = tile_n // subchannel_k
            res_blocked = jnp.reshape(res, (tile_m, num_blocks, subchannel_k))
            abs_max = jnp.max(jnp.abs(res_blocked), axis=-1, keepdims=True)
            scale = jnp.maximum(abs_max / quant_max, 1e-12)
            res_scaled = res_blocked / scale
            res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
            out_q_vmem[out_slot, :, :] = jnp.reshape(
                res_clamped, (tile_m, tile_n)
            ).astype(jnp.float8_e4m3fn)
            vmem_scale[...] = jnp.reshape(scale, (tile_scale_elements,)).astype(
                jnp.bfloat16
            )

    # 0. Initial prefetch for Tile 0
    w_descs_prev = load_weights(0, slot=0)
    store_prev = None
    store_prev2 = None

    for j in range(num_n_tiles):
      slot_curr = j % 2
      slot_next = (j + 1) % 2
      slot_prev = (j - 1) % 2
      out_slot = (j - 1) % 2
      out_prev_slot = (j - 2) % 2

      # 1. Asynchronously dispatch DMA store for Tile j - 2
      if j >= 2:
        wait_store(store_prev2)
        store_prev2 = store_prev
        store_prev = dispatch_store(j - 2, out_slot=out_prev_slot)

      # 2. Asynchronously prefetch weights for Tile j + 1
      if j + 1 < num_n_tiles:
        w_descs_next = load_weights(j + 1, slot=slot_next)

      # 3. Wait for current tile weights W[j]
      wait_weights(w_descs_prev)

      # 4. CONCURRENT EXECUTION:
      # -> MXU: Compute Gate & Up GEMMs for Tile j into acc[slot_curr]
      with jax.named_scope("mxu_gemm_gate_and_up"):
        acc_gate_vmem[slot_curr, :, :] = jnp.dot(
            lhs_tile,
            w_gate_vmem[slot_curr, :, :],
            preferred_element_type=jnp.float32,
        )
        acc_up_vmem[slot_curr, :, :] = jnp.dot(
            lhs_tile,
            w_up_vmem[slot_curr, :, :],
            preferred_element_type=jnp.float32,
        )

      # -> VPU: Concurrently compute dequantization + SwiGLU + FP8 Quant for Tile j - 1
      if j > 0:
        compute_vpu_step(j_prev=j - 1, acc_slot=slot_prev, out_slot=out_slot)

      # 5. Advance weight descriptors
      if j + 1 < num_n_tiles:
        w_descs_prev = w_descs_next

    # Epilogue: Drain pipeline for final remaining tiles
    last_tile = num_n_tiles - 1
    if num_n_tiles >= 2:
      wait_store(store_prev2)
      store_prev2 = store_prev
      store_prev = dispatch_store(
          num_n_tiles - 2, out_slot=(num_n_tiles - 2) % 2
      )

    compute_vpu_step(
        j_prev=last_tile,
        acc_slot=last_tile % 2,
        out_slot=last_tile % 2,
    )

    wait_store(store_prev2)
    wait_store(store_prev)
    store_last = dispatch_store(last_tile, out_slot=last_tile % 2)
    wait_store(store_last)


def _channelwise_quantize_kernel(
    res_ref,
    out_q_ref,
    out_scale_ref,
    *,
    quant_max: float = 448.0,
):
  res = res_ref[...]  # [tile_m, tile_n]
  abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)  # [1, tile_n]
  scale = jnp.maximum(abs_max / quant_max, 1e-12).astype(jnp.bfloat16)
  inv_scale = (1.0 / scale).astype(jnp.bfloat16)
  res_scaled = res * inv_scale
  res_clamped = jnp.clip(res_scaled, -quant_max, quant_max)
  out_q_ref[...] = res_clamped.astype(jnp.float8_e4m3fn)
  out_scale_ref[...] = scale.squeeze(axis=0)


def pallas_channelwise_quantize(
    x_bf16: jax.Array,
    tile_n: int = 512,
    quant_max: float = 448.0,
    vmem_limit_bytes: int = 160 * 1024 * 1024,
    name: str = "pallas_channelwise_quant",
) -> tuple[jax.Array, jax.Array]:
  """Performs single-pass column-wise (channel-wise) FP8 quantization [1, N] via Pallas on TPU."""
  m, n = x_bf16.shape
  tile_m = m
  num_n_tiles = n // tile_n
  grid = (1, num_n_tiles)

  in_spec = pl.BlockSpec((tile_m, tile_n), lambda i, j: (0, j))
  out_specs = [
      pl.BlockSpec((tile_m, tile_n), lambda i, j: (0, j)),
      pl.BlockSpec((tile_n,), lambda i, j: (j,)),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((m, n), jnp.float8_e4m3fn),
      jax.ShapeDtypeStruct((n,), jnp.bfloat16),
  ]

  kernel_fn = functools.partial(
      _channelwise_quantize_kernel,
      quant_max=quant_max,
  )

  out_q, out_scale_flat = pl.pallas_call(
      kernel_fn,
      out_shape=out_shapes,
      grid=grid,
      in_specs=[in_spec],
      out_specs=out_specs,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=name,
  )(x_bf16)

  return out_q, out_scale_flat.reshape((1, n))


def fused_swiglu_pallas(
    lhs_q: jax.Array,
    lhs_scale: jax.Array,
    w_gate_q: jax.Array,
    w_gate_scale: jax.Array,
    w_up_q: jax.Array,
    w_up_scale: jax.Array,
    tile_m: int = 2048,
    tile_n: int = 512,
    tile_k: int = 5120,
    quant_out: bool = False,
    quant_mode: str = "subchannel",
    subchannel_k: int = 512,
    quant_max: float = 448.0,
    pipeline_mode: str = "pipelined",
    vmem_limit_bytes: int = 160 * 1024 * 1024,
    name: str | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Fused SwiGLU Pallas Kernel on TPU with optional Sub-Channel FP8 Quantization.

  Args:
    lhs_q: Input activations [M, K] in float8_e4m3fn.
    lhs_scale: LHS scaling factor [M, 1] or [M, K // subchannel_k_in] in
      bfloat16.
    w_gate_q: Gate weight tensor [K, N] in float8_e4m3fn.
    w_gate_scale: Gate weight scale [1, N] or [K_groups, N] in bfloat16.
    w_up_q: Up weight tensor [K, N] in float8_e4m3fn.
    w_up_scale: Up weight scale [1, N] or [K_groups, N] in bfloat16.
    tile_m: Tile dimension along token sequence M (default: 2048).
    tile_n: Tile dimension along hidden channel N (default: 512).
    tile_k: Tile dimension along contracting K (default: 5120).
    quant_out: If True, dynamically quantizes SwiGLU output to FP8.
    quant_mode: 'subchannel' (in-kernel group-wise [M, N//G]), 'channelwise'
      (in-kernel column-wise [1, N]), 'channelwise_separate_pallas' (separate
      Pallas tiled column-wise reduction [1, N]), 'channelwise_separate_xla'
      (separate XLA column-wise reduction [1, N]), or 'separate'.
    subchannel_k: Sub-channel group size along N for output quantization
      (default: 512).
    quant_max: Maximum FP8 magnitude (default: 448.0 for float8_e4m3fn).
    pipeline_mode: Execution mode ('pipelined' for 3-way decoupled overlap or
      'grid').
    vmem_limit_bytes: VMEM memory allocation limit.
    name: Optional name for the Pallas kernel.

  Returns:
    If quant_out is False:
      out: [M, N] in bfloat16.
    If quant_out is True:
      tuple of (out_q, out_scale) where:
        out_q: [M, N] in float8_e4m3fn.
        out_scale: [M, N // subchannel_k] or [1, N] in bfloat16.
  """
  m, k = lhs_q.shape
  _, n = w_gate_q.shape

  if m % tile_m != 0:
    raise ValueError(f"M ({m}) must be divisible by tile_m ({tile_m})")
  if n % tile_n != 0:
    raise ValueError(f"N ({n}) must be divisible by tile_n ({tile_n})")
  if (
      quant_out
      and (quant_mode == "subchannel")
      and (subchannel_k > 0)
      and (tile_n % subchannel_k != 0)
  ):
    raise ValueError(
        f"tile_n ({tile_n}) must be a multiple of subchannel_k ({subchannel_k})"
    )

  # Support separate quantization pass wrapped inside fused_swiglu_pallas
  if quant_out and quant_mode in (
      "separate",
      "channelwise_separate",
      "channelwise_separate_pallas",
      "channelwise_separate_xla",
      "separate_channelwise",
      "separate_subchannel",
  ):
    res = fused_swiglu_pallas(
        lhs_q=lhs_q,
        lhs_scale=lhs_scale,
        w_gate_q=w_gate_q,
        w_gate_scale=w_gate_scale,
        w_up_q=w_up_q,
        w_up_scale=w_up_scale,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        quant_out=False,
        pipeline_mode=pipeline_mode,
        vmem_limit_bytes=vmem_limit_bytes,
        name=name,
    )
    if quant_mode in ("channelwise_separate", "channelwise_separate_pallas"):
      return pallas_channelwise_quantize(
          res,
          tile_n=tile_n,
          quant_max=quant_max,
          vmem_limit_bytes=vmem_limit_bytes,
          name=f"{name or 'fused_swiglu'}_pallas_cw_quant",
      )
    elif quant_mode in ("channelwise_separate_xla", "separate_channelwise") or (
        quant_mode == "separate" and subchannel_k == 0
    ):
      abs_max = jnp.max(jnp.abs(res), axis=0, keepdims=True)  # [1, N]
      scale = jnp.maximum(abs_max / quant_max, 1e-12).astype(jnp.bfloat16)
      inv_scale = (1.0 / scale).astype(jnp.bfloat16)
      res_scaled = res * inv_scale
      out_q = jnp.clip(res_scaled, -quant_max, quant_max).astype(
          jnp.float8_e4m3fn
      )
      return out_q, scale
    elif subchannel_k <= 0 or subchannel_k >= n:
      abs_max = jnp.max(jnp.abs(res), axis=-1, keepdims=True)  # [M, 1]
      scale = jnp.maximum(abs_max / quant_max, 1e-12).astype(jnp.bfloat16)
      inv_scale = (1.0 / scale).astype(jnp.bfloat16)
      res_scaled = res * inv_scale
      out_q = jnp.clip(res_scaled, -quant_max, quant_max).astype(
          jnp.float8_e4m3fn
      )
      return out_q, scale
    else:
      num_blocks = n // subchannel_k
      res_blocked = jnp.reshape(res, (m, num_blocks, subchannel_k))
      abs_max = jnp.max(jnp.abs(res_blocked), axis=-1, keepdims=True)
      scale = jnp.maximum(abs_max / quant_max, 1e-12).astype(jnp.bfloat16)
      inv_scale = (1.0 / scale).astype(jnp.bfloat16)
      res_scaled = res_blocked * inv_scale
      out_q = jnp.reshape(
          jnp.clip(res_scaled, -quant_max, quant_max), (m, n)
      ).astype(jnp.float8_e4m3fn)
      out_scale = jnp.reshape(scale, (m, num_blocks)).astype(jnp.bfloat16)
      return out_q, out_scale

  if quant_mode == "channelwise":
    tile_scale_elements = tile_n
    num_total_scales = n
  else:
    num_scales_per_tile = (
        1
        if (subchannel_k <= 0 or subchannel_k >= tile_n)
        else (tile_n // subchannel_k)
    )
    num_total_scales = 1 if (subchannel_k <= 0) else (n // subchannel_k)

  if pipeline_mode == "grid":
    num_m_tiles = m // tile_m
    num_n_tiles = n // tile_n
    grid = (num_m_tiles, num_n_tiles)
    in_specs = [
        pl.BlockSpec((tile_m, tile_k), lambda i, j: (i, 0)),
        pl.BlockSpec((tile_m, lhs_scale.shape[1]), lambda i, j: (i, 0)),
        pl.BlockSpec((tile_k, tile_n), lambda i, j: (0, j)),
        pl.BlockSpec((w_gate_scale.shape[0], tile_n), lambda i, j: (0, j)),
        pl.BlockSpec((tile_k, tile_n), lambda i, j: (0, j)),
        pl.BlockSpec((w_up_scale.shape[0], tile_n), lambda i, j: (0, j)),
    ]
    if not quant_out:
      kernel_fn = functools.partial(
          swiglu_kernel_grid,
          tile_m=tile_m,
          tile_n=tile_n,
          tile_k=tile_k,
          quant_out=False,
      )
      out_spec = pl.BlockSpec((tile_m, tile_n), lambda i, j: (i, j))
      out_shape = jax.ShapeDtypeStruct((m, n), jnp.bfloat16)
      return pl.pallas_call(
          kernel_fn,
          out_shape=out_shape,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_spec,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=vmem_limit_bytes,
              disable_bounds_checks=True,
          ),
          name=name or "fused_swiglu_grid",
      )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

    kernel_fn = functools.partial(
        swiglu_kernel_grid,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        quant_out=True,
        quant_mode=quant_mode,
        subchannel_k=subchannel_k,
        quant_max=quant_max,
    )
    if quant_mode == "channelwise":
      out_specs = [
          pl.BlockSpec((tile_m, tile_n), lambda i, j: (i, j)),
          pl.BlockSpec((tile_n,), lambda i, j: (i * num_n_tiles + j,)),
      ]
      out_shapes = [
          jax.ShapeDtypeStruct((m, n), jnp.float8_e4m3fn),
          jax.ShapeDtypeStruct((num_m_tiles * n,), jnp.bfloat16),
      ]
      out_q, out_scale_flat = pl.pallas_call(
          kernel_fn,
          out_shape=out_shapes,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=vmem_limit_bytes,
              disable_bounds_checks=True,
          ),
          name=name or "fused_swiglu_grid_quant_cw",
      )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)
      if num_m_tiles == 1:
        out_scale = out_scale_flat.reshape((1, n))
      else:
        out_scale = out_scale_flat.reshape((num_m_tiles, n))
      return out_q, out_scale
    else:
      num_sub_k = (
          1
          if (subchannel_k <= 0 or subchannel_k >= tile_n)
          else (tile_n // subchannel_k)
      )
      tile_scale_elements = tile_m * num_sub_k
      out_specs = [
          pl.BlockSpec((tile_m, tile_n), lambda i, j: (i, j)),
          pl.BlockSpec(
              (tile_scale_elements,), lambda i, j: (i * num_n_tiles + j,)
          ),
      ]
      out_shapes = [
          jax.ShapeDtypeStruct((m, n), jnp.float8_e4m3fn),
          jax.ShapeDtypeStruct((m * num_total_scales,), jnp.bfloat16),
      ]
      out_q, out_scale_flat = pl.pallas_call(
          kernel_fn,
          out_shape=out_shapes,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          compiler_params=pltpu.CompilerParams(
              vmem_limit_bytes=vmem_limit_bytes,
              disable_bounds_checks=True,
          ),
          name=name or f"fused_swiglu_grid_quant_subchannel_g{subchannel_k}",
      )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

      if num_sub_k == 1:
        out_scale = (
            out_scale_flat.reshape((num_m_tiles, num_n_tiles, tile_m))
            .transpose(0, 2, 1)
            .reshape((m, num_total_scales))
        )
      else:
        out_scale = (
            out_scale_flat.reshape(
                (num_m_tiles, num_n_tiles, tile_m, num_sub_k)
            )
            .transpose(0, 2, 1, 3)
            .reshape((m, num_total_scales))
        )
      return out_q, out_scale

  # 3-Way Decoupled Software Pipeline with Stationary LHS
  num_m_tiles = m // tile_m
  num_n_tiles = n // tile_n

  in_specs_pipe = [
      pl.BlockSpec((tile_m, tile_k), lambda i: (i, 0)),
      pl.BlockSpec((tile_m, lhs_scale.shape[1]), lambda i: (i, 0)),
      pl.BlockSpec(memory_space=pltpu.HBM),
      pl.BlockSpec(memory_space=pltpu.HBM),
      pl.BlockSpec(memory_space=pltpu.HBM),
      pl.BlockSpec(memory_space=pltpu.HBM),
  ]

  num_sems = 10 if not quant_out else 12
  kernel_fn = functools.partial(
      _pipelined_swiglu_kernel,
      tile_m=tile_m,
      tile_n=tile_n,
      tile_k=tile_k,
      num_n_tiles=num_n_tiles,
      quant_out=quant_out,
      quant_mode=quant_mode,
      subchannel_k=subchannel_k,
      quant_max=quant_max,
  )

  kernel_name = name or (
      f"pipelined_swiglu_m{tile_m}_n{tile_n}_k{tile_k}"
      + ("_quant" if quant_out else "")
  )

  if not quant_out:
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=in_specs_pipe,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        grid=(num_m_tiles,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * num_sems,
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=kernel_name,
    )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

  grid_spec = pltpu.PrefetchScalarGridSpec(
      num_scalar_prefetch=0,
      in_specs=in_specs_pipe,
      out_specs=[
          pl.BlockSpec(memory_space=pltpu.HBM),
          pl.BlockSpec(memory_space=pltpu.HBM),
      ],
      grid=(num_m_tiles,),
      scratch_shapes=[pltpu.SemaphoreType.DMA] * num_sems,
  )
  if quant_mode == "channelwise":
    scale_flat_len = n if num_m_tiles == 1 else num_m_tiles * n
  else:
    scale_flat_len = m * num_total_scales

  out_shapes = [
      jax.ShapeDtypeStruct((m, n), jnp.float8_e4m3fn),
      jax.ShapeDtypeStruct((scale_flat_len,), jnp.bfloat16),
  ]
  out_q, out_scale_flat = pl.pallas_call(
      kernel_fn,
      out_shape=out_shapes,
      grid_spec=grid_spec,
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      name=kernel_name,
  )(lhs_q, lhs_scale, w_gate_q, w_gate_scale, w_up_q, w_up_scale)

  if quant_mode == "channelwise":
    if num_m_tiles == 1:
      out_scale = out_scale_flat.reshape((1, n))
    else:
      out_scale = out_scale_flat.reshape((num_m_tiles, n))
  else:
    if num_scales_per_tile == 1:
      out_scale = (
          out_scale_flat.reshape((num_m_tiles, num_n_tiles, tile_m))
          .transpose(0, 2, 1)
          .reshape((m, num_total_scales))
      )
    else:
      out_scale = (
          out_scale_flat.reshape(
              (num_m_tiles, num_n_tiles, tile_m, num_scales_per_tile)
          )
          .transpose(0, 2, 1, 3)
          .reshape((m, num_total_scales))
      )
  return out_q, out_scale
