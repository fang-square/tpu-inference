"""Fused All-Reduce MatMul Pallas Kernel on TPU supporting FP8 (f8e4m3fn) and BF16."""

from collections.abc import Sequence
import functools
import math
from typing import Any, Tuple, Union

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp

P = jax.sharding.PartitionSpec


def _fused_all_reduce_matmul_kernel_5stage_tp2(
    x_hbm_ref,
    w_hbm_ref,
    o_hbm_ref,
    *sems,
    block_m: int,
    block_n: int,
    block_k: int,
    axis_name: str = "tp",
    mesh_axes: Sequence[str] = ("tp",),
    k_local: int | None = None,
):
  """5-Stage Fully Decoupled Hardware Pipeline for TP=2 Fused All-Reduce MatMul:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): ICI/D2D Remote DMA Send/Recv in flight across shards.
  - Stage 3 (Tile k-2): VPU Vector Addition & Asynchronous VMEM -> HBM DMA
  start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index(axis_name)
  peer_id = 1 - my_id

  hbm_x_sems = sems[0:2]
  hbm_w_sems = sems[2:4]
  hbm_out_sems = sems[4:6]
  remote_send_sems = sems[6:9]
  remote_recv_sems = sems[9:12]

  if k_local is None:
    k_local = x_hbm_ref.shape[1]
  num_k = k_local // block_k

  @functools.partial(
      pl.run_scoped,
      x_vmem=pltpu.VMEM((2, block_m, block_k), x_hbm_ref.dtype),
      w_vmem=pltpu.VMEM((2, block_k, block_n), w_hbm_ref.dtype),
      acc_vmem=pltpu.VMEM((3, block_m, block_n), o_hbm_ref.dtype),
      recv_vmem=pltpu.VMEM((3, block_m, block_n), o_hbm_ref.dtype),
      out_vmem=pltpu.VMEM((2, block_m, block_n), o_hbm_ref.dtype),
  )
  def _(x_vmem, w_vmem, acc_vmem, recv_vmem, out_vmem):

    def get_tile_slices(tile_idx):
      i = tile_idx // num_n
      j = tile_idx % num_n
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      if x_hbm_ref.shape[1] > k_local:
        # Native Strided DMA directly from global un-sharded tensor X [M, K_total]
        k_start = my_id * k_local + k_idx * block_k
      else:
        # Pre-sharded tensor X [M, K_local]
        k_start = k_idx * block_k
      k_slice = pl.ds(k_start, block_k)

      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sems[slot],
        )
        copy_x.start()

      if w_hbm_ref.shape[0] > k_local:
        # Native Strided DMA directly from global un-sharded weight W [K_total, N]
        w_k_start = my_id * k_local + k_idx * block_k
      else:
        # Pre-sharded weight W [K_local, N]
        w_k_start = k_idx * block_k
      w_k_slice = pl.ds(w_k_start, block_k)

      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[w_k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sems[slot],
        )
        copy_w.start()
      return copy_x, copy_w

    def dispatch_network(acc_slot):
      with jax.named_scope("all_reduce_exchange"):
        # if len(mesh_axes) == 1:
        #   dest_dev = (peer_id,)
        # else:
        #   dest_dev = tuple(
        #       lax.axis_index(ax) if ax != axis_name else peer_id
        #       for ax in mesh_axes
        #   )
        copy_desc = pltpu.make_async_remote_copy(
            src_ref=acc_vmem.at[acc_slot, :, :],
            dst_ref=recv_vmem.at[acc_slot, :, :],
            send_sem=remote_send_sems[acc_slot],
            recv_sem=remote_recv_sems[acc_slot],
            device_id={axis_name:peer_id},
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy_desc.start()
        return copy_desc

    def vpu_add_and_dispatch_hbm_store(
        tile_idx, acc_slot, out_slot, net_copy_desc
    ):
      with jax.named_scope("all_reduce_vpu_add"):
        net_copy_desc.wait()
        out_vmem[out_slot, :, :] = (
            acc_vmem[acc_slot, :, :] + recv_vmem[acc_slot, :, :]
        )
        m_slice, n_slice = get_tile_slices(tile_idx)
        hbm_store_desc = pltpu.make_async_copy(
            src_ref=out_vmem.at[out_slot, :, :],
            dst_ref=o_hbm_ref.at[m_slice, n_slice],
            sem=hbm_out_sems[out_slot],
        )
        hbm_store_desc.start()
        return hbm_store_desc

    def wait_hbm_store(hbm_store_desc):
      with jax.named_scope("hbm_wait_store"):
        hbm_store_desc.wait()

    total_chunks = num_tiles * num_k

    def load_chunk(chunk_idx, slot):
      t = chunk_idx // num_k
      ki = chunk_idx % num_k
      return load_hbm_inputs(t, slot=slot, k_idx=ki)

    # Initial prefetch: load chunk 0 into slot 0
    cx_prev, cw_prev = load_chunk(0, slot=0)

    # Track in-flight network copy and HBM store descriptors
    copy_descs = [None, None]
    store_prev = None

    for c in range(total_chunks):
      t = c // num_k
      ki = c % num_k
      in_curr = c % 2
      in_next = 1 - in_curr
      acc_curr = t % 3

      # 1. Asynchronously prefetch next chunk c + 1 into alternate input slot
      if c + 1 < total_chunks:
        cx_next, cw_next = load_chunk(c + 1, slot=in_next)

      # 2. Wait for current chunk inputs
      with jax.named_scope("hbm_wait_inputs"):
        cx_prev.wait()
        cw_prev.wait()

      # 3. Compute MXU MatMul for chunk c
      with jax.named_scope("matmul_running"):
        sub_dot = jnp.dot(
            x_vmem[in_curr, :, :],
            w_vmem[in_curr, :, :],
            preferred_element_type=jnp.float32,
        )
        if ki == 0:
          acc_vmem[acc_curr, :, :] = sub_dot.astype(o_hbm_ref.dtype)
        else:
          acc_vmem[acc_curr, :, :] = acc_vmem[acc_curr, :, :] + sub_dot.astype(
              o_hbm_ref.dtype
          )

      # 4. Advance input prefetch descriptors
      if c + 1 < total_chunks:
        cx_prev = cx_next
        cw_prev = cw_next

      # 5. When spatial tile t completes its last K chunk (ki == num_k - 1):
      if ki == num_k - 1:
        # Dispatch network remote copy for tile t
        copy_curr = dispatch_network(acc_slot=acc_curr)

        if t == 0:
          copy_descs[0] = copy_curr
        elif t == 1:
          copy_descs[1] = copy_curr
          # For tile 0: start VPU add and HBM store
          store_prev = vpu_add_and_dispatch_hbm_store(
              0, acc_slot=0, out_slot=0, net_copy_desc=copy_descs[0]
          )
        else:
          # For tile t >= 2:
          # Wait for previous HBM store of tile t - 2 to finish
          wait_hbm_store(store_prev)
          # Start HBM store for tile t - 1
          acc_prev1 = (t - 1) % 3
          out_prev1 = (t - 1) % 2
          store_prev = vpu_add_and_dispatch_hbm_store(
              t - 1,
              acc_slot=acc_prev1,
              out_slot=out_prev1,
              net_copy_desc=copy_descs[1],
          )
          copy_descs[0] = copy_descs[1]
          copy_descs[1] = copy_curr

    # Epilogue: retire remaining in-flight tiles
    if num_tiles == 1:
      store_last = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_desc=copy_descs[0]
      )
      wait_hbm_store(store_last)
    elif num_tiles == 2:
      wait_hbm_store(store_prev)
      store_last = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_desc=copy_descs[1]
      )
      wait_hbm_store(store_last)
    else:  # num_tiles >= 3
      wait_hbm_store(store_prev)
      store_last = vpu_add_and_dispatch_hbm_store(
          num_tiles - 1,
          acc_slot=(num_tiles - 1) % 3,
          out_slot=(num_tiles - 1) % 2,
          net_copy_desc=copy_descs[1],
      )
      wait_hbm_store(store_last)


def _fused_all_reduce_matmul_kernel_5stage_ring(
    x_hbm_ref,
    w_hbm_ref,
    o_hbm_ref,
    *sems,
    num_devices: int,
    block_m: int,
    block_n: int,
    block_k: int,
    k_local: int | None = None,
):
  """5-Stage Decoupled Pipeline for TP >= 2 Ring All-Reduce MatMul:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): Ring Network Remote DMA Send/Recv in flight across
  chips.
  - Stage 3 (Tile k-2): VPU Vector Addition & Asynchronous VMEM -> HBM DMA
  start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index("tp")
  right_neighbor = lax.rem(my_id + 1, num_devices)

  num_rounds = num_devices - 1
  hbm_x_sems = sems[0:2]
  hbm_w_sems = sems[2:4]
  hbm_out_sems = sems[4:6]
  remote_send_sems = sems[6 : 6 + num_rounds]
  remote_recv_sems = sems[6 + num_rounds : 6 + 2 * num_rounds]

  if k_local is None:
    k_local = x_hbm_ref.shape[1]
  num_k = k_local // block_k

  @functools.partial(
      pl.run_scoped,
      x_vmem=pltpu.VMEM((2, block_m, block_k), x_hbm_ref.dtype),
      w_vmem=pltpu.VMEM((2, block_k, block_n), w_hbm_ref.dtype),
      acc_vmem=pltpu.VMEM((3, block_m, block_n), o_hbm_ref.dtype),
      recv_vmem=pltpu.VMEM((3, block_m, block_n), o_hbm_ref.dtype),
      ring_scratch=pltpu.VMEM((2, block_m, block_n), o_hbm_ref.dtype),
      out_vmem=pltpu.VMEM((2, block_m, block_n), o_hbm_ref.dtype),
  )
  def _(x_vmem, w_vmem, acc_vmem, recv_vmem, ring_scratch, out_vmem):

    def get_tile_slices(tile_idx):
      i = tile_idx // num_n
      j = tile_idx % num_n
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      if x_hbm_ref.shape[1] > k_local:
        k_start = my_id * k_local + k_idx * block_k
      else:
        k_start = k_idx * block_k
      k_slice = pl.ds(k_start, block_k)

      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sems[slot],
        )
        copy_x.start()

      if w_hbm_ref.shape[0] > k_local:
        w_k_start = my_id * k_local + k_idx * block_k
      else:
        w_k_start = k_idx * block_k
      w_k_slice = pl.ds(w_k_start, block_k)

      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[w_k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sems[slot],
        )
        copy_w.start()
      return copy_x, copy_w

    def compute_tile(tile_idx, in_slot, acc_slot, copy_x, copy_w):
      with jax.named_scope("hbm_wait_inputs"):
        copy_x.wait()
        copy_w.wait()
      with jax.named_scope("matmul_running"):
        sub_acc = jnp.dot(
            x_vmem[in_slot, :, :],
            w_vmem[in_slot, :, :],
            preferred_element_type=jnp.float32,
        )
        acc_vmem[acc_slot, :, :] = sub_acc.astype(o_hbm_ref.dtype)
      for ki in range(1, num_k):
        cx_k, cw_k = load_hbm_inputs(tile_idx, slot=in_slot, k_idx=ki)
        with jax.named_scope("hbm_wait_inputs"):
          cx_k.wait()
          cw_k.wait()
        with jax.named_scope("matmul_running"):
          sub_acc_k = jnp.dot(
              x_vmem[in_slot, :, :],
              w_vmem[in_slot, :, :],
              preferred_element_type=jnp.float32,
          )
          acc_vmem[acc_slot, :, :] = acc_vmem[
              acc_slot, :, :
          ] + sub_acc_k.astype(o_hbm_ref.dtype)

    def dispatch_network(acc_slot):
      with jax.named_scope("all_reduce_ring_exchange"):
        ring_scratch[0, :, :] = acc_vmem[acc_slot, :, :]
        recv_vmem[acc_slot, :, :] = acc_vmem[acc_slot, :, :]

        for step in range(num_rounds):
          src_slot = step % 2
          dst_slot = 1 - src_slot
          copy_desc = pltpu.make_async_remote_copy(
              src_ref=ring_scratch.at[src_slot, :, :],
              dst_ref=ring_scratch.at[dst_slot, :, :],
              send_sem=remote_send_sems[step],
              recv_sem=remote_recv_sems[step],
              device_id=(right_neighbor,),
              device_id_type=pl.DeviceIdType.MESH,
          )
          copy_desc.start()
          copy_desc.wait()
          recv_vmem[acc_slot, :, :] += ring_scratch[dst_slot, :, :]
        return None

    def vpu_add_and_dispatch_hbm_store(
        tile_idx, acc_slot, out_slot, net_copy_desc
    ):
      with jax.named_scope("all_reduce_vpu_store"):
        out_vmem[out_slot, :, :] = recv_vmem[acc_slot, :, :]
        m_slice, n_slice = get_tile_slices(tile_idx)
        hbm_store_desc = pltpu.make_async_copy(
            src_ref=out_vmem.at[out_slot, :, :],
            dst_ref=o_hbm_ref.at[m_slice, n_slice],
            sem=hbm_out_sems[out_slot],
        )
        hbm_store_desc.start()
        return hbm_store_desc

    def wait_hbm_store(hbm_store_desc):
      with jax.named_scope("hbm_wait_store"):
        hbm_store_desc.wait()

    if num_tiles == 1:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_desc=copy_0
      )
      wait_hbm_store(store_0)
    elif num_tiles == 2:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_desc=copy_0
      )
      store_1 = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_desc=copy_1
      )
      wait_hbm_store(store_0)
      wait_hbm_store(store_1)
    elif num_tiles == 3:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      cx2, cw2 = load_hbm_inputs(2, slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_desc=copy_0
      )
      compute_tile(2, in_slot=0, acc_slot=2, copy_x=cx2, copy_w=cw2)
      copy_2 = dispatch_network(acc_slot=2)
      wait_hbm_store(store_0)
      store_1 = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_desc=copy_1
      )
      store_2 = vpu_add_and_dispatch_hbm_store(
          2, acc_slot=2, out_slot=0, net_copy_desc=copy_2
      )
      wait_hbm_store(store_1)
      wait_hbm_store(store_2)
    else:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_prev2 = dispatch_network(acc_slot=0)
      cx2, cw2 = load_hbm_inputs(2, slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_prev1 = dispatch_network(acc_slot=1)
      store_prev1 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_desc=copy_prev2
      )
      cx3, cw3 = load_hbm_inputs(3, slot=1)
      compute_tile(2, in_slot=0, acc_slot=2, copy_x=cx2, copy_w=cw2)
      copy_prev2 = copy_prev1
      copy_prev1 = dispatch_network(acc_slot=2)

      cx_prev = cx3
      cw_prev = cw3

      for k in range(3, num_tiles):
        in_curr = lax.rem(k, 2)
        in_next = 1 - in_curr
        acc_curr = lax.rem(k, 3)
        acc_prev2 = lax.rem(k - 2, 3)
        out_curr = lax.rem(k - 2, 2)

        wait_hbm_store(store_prev1)
        store_prev1 = vpu_add_and_dispatch_hbm_store(
            k - 2,
            acc_slot=acc_prev2,
            out_slot=out_curr,
            net_copy_desc=copy_prev2,
        )

        if k + 1 < num_tiles:
          cx_next, cw_next = load_hbm_inputs(k + 1, slot=in_next)

        compute_tile(
            k,
            in_slot=in_curr,
            acc_slot=acc_curr,
            copy_x=cx_prev,
            copy_w=cw_prev,
        )

        if k + 1 < num_tiles:
          cx_prev = cx_next
          cw_prev = cw_next

        copy_prev2 = copy_prev1
        copy_prev1 = dispatch_network(acc_slot=acc_curr)

      wait_hbm_store(store_prev1)
      store_penultimate = vpu_add_and_dispatch_hbm_store(
          num_tiles - 2,
          acc_slot=lax.rem(num_tiles - 2, 3),
          out_slot=lax.rem(num_tiles - 2, 2),
          net_copy_desc=copy_prev2,
      )
      store_last = vpu_add_and_dispatch_hbm_store(
          num_tiles - 1,
          acc_slot=lax.rem(num_tiles - 1, 3),
          out_slot=lax.rem(num_tiles - 1, 2),
          net_copy_desc=copy_prev1,
      )
      wait_hbm_store(store_penultimate)
      wait_hbm_store(store_last)


def _fused_all_reduce_matmul_kernel_5stage_all2all(
    x_hbm_ref,
    w_hbm_ref,
    o_hbm_ref,
    *sems,
    num_devices: int,
    block_m: int,
    block_n: int,
    block_k: int,
    k_local: int | None = None,
):
  """5-Stage Decoupled Pipeline for TP >= 2 All-to-All All-Reduce MatMul on Sunfish:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): Simultaneous 1-to-(TP-1) point-to-point remote DMA in
  flight.
  - Stage 3 (Tile k-2): VPU multi-peer accumulation & Asynchronous VMEM -> HBM
  DMA start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index("tp")
  num_peers = num_devices - 1
  hbm_x_sems = sems[0:2]
  hbm_w_sems = sems[2:4]
  hbm_out_sems = sems[4:6]
  remote_send_sems = sems[6 : 6 + num_peers]
  remote_recv_sems = sems[6 + num_peers : 6 + 2 * num_peers]

  if k_local is None:
    k_local = x_hbm_ref.shape[1]
  num_k = k_local // block_k

  @functools.partial(
      pl.run_scoped,
      x_vmem=pltpu.VMEM((2, block_m, block_k), x_hbm_ref.dtype),
      w_vmem=pltpu.VMEM((2, block_k, block_n), w_hbm_ref.dtype),
      acc_vmem=pltpu.VMEM((3, block_m, block_n), o_hbm_ref.dtype),
      recv_vmem=pltpu.VMEM((num_peers, 3, block_m, block_n), o_hbm_ref.dtype),
      out_vmem=pltpu.VMEM((2, block_m, block_n), o_hbm_ref.dtype),
  )
  def _(x_vmem, w_vmem, acc_vmem, recv_vmem, out_vmem):

    def get_tile_slices(tile_idx):
      i = tile_idx // num_n
      j = tile_idx % num_n
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      if x_hbm_ref.shape[1] > k_local:
        k_start = my_id * k_local + k_idx * block_k
      else:
        k_start = k_idx * block_k
      k_slice = pl.ds(k_start, block_k)

      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sems[slot],
        )
        copy_x.start()

      if w_hbm_ref.shape[0] > k_local:
        w_k_start = my_id * k_local + k_idx * block_k
      else:
        w_k_start = k_idx * block_k
      w_k_slice = pl.ds(w_k_start, block_k)

      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[w_k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sems[slot],
        )
        copy_w.start()
      return copy_x, copy_w

    def compute_tile(tile_idx, in_slot, acc_slot, copy_x, copy_w):
      with jax.named_scope("hbm_wait_inputs"):
        copy_x.wait()
        copy_w.wait()
      with jax.named_scope("matmul_running"):
        sub_acc = jnp.dot(
            x_vmem[in_slot, :, :],
            w_vmem[in_slot, :, :],
            preferred_element_type=jnp.float32,
        )
        acc_vmem[acc_slot, :, :] = sub_acc.astype(o_hbm_ref.dtype)
      for ki in range(1, num_k):
        cx_k, cw_k = load_hbm_inputs(tile_idx, slot=in_slot, k_idx=ki)
        with jax.named_scope("hbm_wait_inputs"):
          cx_k.wait()
          cw_k.wait()
        with jax.named_scope("matmul_running"):
          sub_acc_k = jnp.dot(
              x_vmem[in_slot, :, :],
              w_vmem[in_slot, :, :],
              preferred_element_type=jnp.float32,
          )
          acc_vmem[acc_slot, :, :] = acc_vmem[
              acc_slot, :, :
          ] + sub_acc_k.astype(o_hbm_ref.dtype)

    def dispatch_network(acc_slot):
      with jax.named_scope("all_reduce_all2all_exchange"):
        descs = []
        for p in range(num_peers):
          target_id = (my_id + p + 1) % num_devices
          copy_desc = pltpu.make_async_remote_copy(
              src_ref=acc_vmem.at[acc_slot, :, :],
              dst_ref=recv_vmem.at[p, acc_slot, :, :],
              send_sem=remote_send_sems[p],
              recv_sem=remote_recv_sems[p],
              device_id=(target_id,),
              device_id_type=pl.DeviceIdType.MESH,
          )
          copy_desc.start()
          descs.append(copy_desc)
        return descs

    def vpu_add_and_dispatch_hbm_store(
        tile_idx, acc_slot, out_slot, net_copy_descs
    ):
      with jax.named_scope("all_reduce_vpu_add_all2all"):
        if net_copy_descs is not None:
          for desc in net_copy_descs:
            desc.wait()
        acc_sum = acc_vmem[acc_slot, :, :]
        for p in range(num_peers):
          acc_sum += recv_vmem[p, acc_slot, :, :]
        out_vmem[out_slot, :, :] = acc_sum
        m_slice, n_slice = get_tile_slices(tile_idx)
        hbm_store_desc = pltpu.make_async_copy(
            src_ref=out_vmem.at[out_slot, :, :],
            dst_ref=o_hbm_ref.at[m_slice, n_slice],
            sem=hbm_out_sems[out_slot],
        )
        hbm_store_desc.start()
        return hbm_store_desc

    def wait_hbm_store(hbm_store_desc):
      with jax.named_scope("hbm_wait_store"):
        hbm_store_desc.wait()

    if num_tiles == 1:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_0
      )
      wait_hbm_store(store_0)
    elif num_tiles == 2:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_0
      )
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_1 = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_descs=copy_1
      )
      wait_hbm_store(store_0)
      wait_hbm_store(store_1)
    elif num_tiles == 3:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_0 = dispatch_network(acc_slot=0)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_0
      )
      cx2, cw2 = load_hbm_inputs(2, slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_1 = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_descs=copy_1
      )
      compute_tile(2, in_slot=0, acc_slot=2, copy_x=cx2, copy_w=cw2)
      copy_2 = dispatch_network(acc_slot=2)
      store_2 = vpu_add_and_dispatch_hbm_store(
          2, acc_slot=2, out_slot=0, net_copy_descs=copy_2
      )
      wait_hbm_store(store_0)
      wait_hbm_store(store_1)
      wait_hbm_store(store_2)
    else:
      cx0, cw0 = load_hbm_inputs(0, slot=0)
      cx1, cw1 = load_hbm_inputs(1, slot=1)
      compute_tile(0, in_slot=0, acc_slot=0, copy_x=cx0, copy_w=cw0)
      copy_prev2 = dispatch_network(acc_slot=0)
      cx2, cw2 = load_hbm_inputs(2, slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_prev1 = dispatch_network(acc_slot=1)
      store_prev1 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_prev2
      )
      cx3, cw3 = load_hbm_inputs(3, slot=1)
      compute_tile(2, in_slot=0, acc_slot=2, copy_x=cx2, copy_w=cw2)
      copy_prev2 = copy_prev1
      copy_prev1 = dispatch_network(acc_slot=2)

      cx_prev = cx3
      cw_prev = cw3

      for k in range(3, num_tiles):
        in_curr = lax.rem(k, 2)
        in_next = 1 - in_curr
        acc_curr = lax.rem(k, 3)
        acc_prev2 = lax.rem(k - 2, 3)
        out_curr = lax.rem(k - 2, 2)

        wait_hbm_store(store_prev1)
        store_prev1 = vpu_add_and_dispatch_hbm_store(
            k - 2,
            acc_slot=acc_prev2,
            out_slot=out_curr,
            net_copy_descs=copy_prev2,
        )

        if k + 1 < num_tiles:
          cx_next, cw_next = load_hbm_inputs(k + 1, slot=in_next)

        compute_tile(
            k,
            in_slot=in_curr,
            acc_slot=acc_curr,
            copy_x=cx_prev,
            copy_w=cw_prev,
        )

        if k + 1 < num_tiles:
          cx_prev = cx_next
          cw_prev = cw_next

        copy_prev2 = copy_prev1
        copy_prev1 = dispatch_network(acc_slot=acc_curr)

      wait_hbm_store(store_prev1)
      store_penultimate = vpu_add_and_dispatch_hbm_store(
          num_tiles - 2,
          acc_slot=lax.rem(num_tiles - 2, 3),
          out_slot=lax.rem(num_tiles - 2, 2),
          net_copy_descs=copy_prev2,
      )
      store_last = vpu_add_and_dispatch_hbm_store(
          num_tiles - 1,
          acc_slot=lax.rem(num_tiles - 1, 3),
          out_slot=lax.rem(num_tiles - 1, 2),
          net_copy_descs=copy_prev1,
      )
      wait_hbm_store(store_penultimate)
      wait_hbm_store(store_last)


def fused_all_reduce_matmul(
    x: jax.Array,
    w: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str = "tp",
    out_dtype: jnp.dtype = jnp.bfloat16,
    block_m: int = 1024,
    block_n: int = 1024,
    block_k: int = 0,
    k_local: int | None = None,
    compute_m: int | None = None,
    compute_n: int | None = None,
    pipeline_mode: str = "5stage",
    interpret: Union[bool, pltpu.InterpretParams, None] = None,
    strided_dma: bool = True,
) -> jax.Array:
  """Executes fused MatMul and All-Reduce across the provided mesh.

  Always utilizes a 5-stage decoupled software pipeline across all topologies:
  - TP=2: Direct 1-hop D2D/ICI exchange.
  - TP>=2 (All2All): 1-hop simultaneous point-to-point dispatches (Sunfish
  Boardfly).
  - TP>=2 (Ring): (TP-1) ring rounds with 2D Intra-D2D + ICI overlap
  (Ghostfish/Zebrafish).

  Args:
    x: Input tensor [M, K_local] (if sharded) or [M, K_total] (if global
      un-sharded with strided DMA).
    w: Weight tensor [K_local, N] (if sharded) or [K_total, N] (if global).
    mesh: JAX device mesh containing the axis_name.
    axis_name: Name of the mesh axis representing Tensor Parallelism.
    out_dtype: Output and All-Reduce reduction dtype (default: jnp.bfloat16).
    block_m: Tiling block size along M dimension.
    block_n: Tiling block size along N dimension.
    block_k: Tiling block size along contracting K dimension.
    k_local: Optional explicit local K dimension per device.
    compute_m: Optional internal compute sub-tile size along M.
    compute_n: Optional internal compute sub-tile size along N.
    pipeline_mode: Collective scheduling mode ("5stage", "all2all", or "ring").
    interpret: Whether to run in Pallas interpret mode.
    strided_dma: If True, uses native strided DMA from global inputs without JAX
      shard_map slicing.

  Returns:
    Fully reduced output tensor [M, N] in out_dtype (BF16) replicated across
    shards.
  """
  m = x.shape[0]
  x_k = x.shape[1]
  w_k = w.shape[0]
  n = w.shape[1]

  num_devices = mesh.shape[axis_name]

  if k_local is None:
    if x_k % num_devices == 0:
      k_local = x_k // num_devices
    else:
      k_local = x_k

  block_m = min(block_m, m)
  block_n = min(block_n, n)
  if block_k == 0:
    block_k = k_local
  else:
    block_k = min(block_k, k_local)

  if interpret is None:
    interpret = (
        pltpu.InterpretParams() if jax.default_backend() == "cpu" else False
    )

  mesh_axes = tuple(mesh.axis_names)

  if num_devices == 2:
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 12,
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_tp2,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        axis_name=axis_name,
        mesh_axes=mesh_axes,
        k_local=k_local,
    )

    if k_local != block_k:
      kernel_name = f"fused_matmul_ar_5stage_tp2_m{block_m}_n{block_n}_k{block_k}_klocal{k_local}"
    else:
      kernel_name = f"fused_matmul_ar_5stage_tp2_m{block_m}_n{block_n}_k{block_k}"
    pallas_fn = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(),
        interpret=interpret,
        name=kernel_name,
    )

    def _shard_fn(x_shard, w_shard):
      with jax.named_scope(kernel_name):
        return pallas_fn(x_shard, w_shard)

  elif pipeline_mode == "all2all":
    num_peers = num_devices - 1
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * (6 + 2 * num_peers),
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_all2all,
        num_devices=num_devices,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        k_local=k_local,
    )

    if k_local != block_k:
      kernel_name = f"fused_matmul_ar_5stage_all2all_m{block_m}_n{block_n}_k{block_k}_klocal{k_local}"
    else:
      kernel_name = f"fused_matmul_ar_5stage_all2all_m{block_m}_n{block_n}_k{block_k}"
    pallas_fn = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(),
        interpret=interpret,
        name=kernel_name,
    )

    def _shard_fn(x_shard, w_shard):
      with jax.named_scope(kernel_name):
        return pallas_fn(x_shard, w_shard)

  else:  # ring mode (default for TP >= 2 on GF, ZF, and ring topologies)
    num_rounds = num_devices - 1
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * (6 + 2 * num_rounds),
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_ring,
        num_devices=num_devices,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        k_local=k_local,
    )

    if k_local != block_k:
      kernel_name = f"fused_matmul_ar_5stage_ring_m{block_m}_n{block_n}_k{block_k}_klocal{k_local}"
    else:
      kernel_name = f"fused_matmul_ar_5stage_ring_m{block_m}_n{block_n}_k{block_k}"
    pallas_fn = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(),
        interpret=interpret,
        name=kernel_name,
    )

    def _shard_fn(x_shard, w_shard):
      with jax.named_scope(kernel_name):
        return pallas_fn(x_shard, w_shard)

  in_specs = (P(None, axis_name), P(axis_name, None))

  outer_name = f"fused_matmul_ar_m{block_m}_n{block_n}_k{block_k}" if k_local == block_k else f"fused_matmul_ar_m{block_m}_n{block_n}_k{block_k}_klocal{k_local}"
  with jax.named_scope(outer_name):
    return shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(None, None),
        check_rep=False,
    )(x, w)
