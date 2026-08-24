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
    hbm_x_sem,
    hbm_w_sem,
    hbm_out_sem,
    remote_send_sem,
    remote_recv_sem,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    axis_name: str = "tp",
):
  """5-Stage Fully Decoupled Hardware Pipeline for TP=2 Fused All-Reduce MatMul:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): ICI/D2D Remote DMA Send/Recv in flight across shards.
  - Stage 3 (Tile k-2): VPU Vector Addition & Asynchronous VMEM -> HBM DMA start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index(axis_name)
  peer_id = 1 - my_id

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
      j = lax.rem(tile_idx, num_n)
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      k_slice = pl.ds(k_idx * block_k, block_k)
      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sem,
        )
        copy_x.start()
      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sem,
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
          acc_vmem[acc_slot, :, :] = (
              acc_vmem[acc_slot, :, :] + sub_acc_k.astype(o_hbm_ref.dtype)
          )

    def dispatch_network(acc_slot):
      with jax.named_scope("all_reduce_exchange"):
        copy_desc = pltpu.make_async_remote_copy(
            src_ref=acc_vmem.at[acc_slot, :, :],
            dst_ref=recv_vmem.at[acc_slot, :, :],
            send_sem=remote_send_sem,
            recv_sem=remote_recv_sem,
            device_id={axis_name: peer_id},
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
            sem=hbm_out_sem,
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
      # --- Full 5-Stage Hardware Pipeline (num_tiles >= 4) ---
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


def _fused_all_reduce_matmul_kernel_5stage_ring(
    x_hbm_ref,
    w_hbm_ref,
    o_hbm_ref,
    hbm_x_sem,
    hbm_w_sem,
    hbm_out_sem,
    *network_sems,
    num_devices: int,
    block_m: int,
    block_n: int,
    block_k: int,
    axis_name: str = "tp",
):
  """5-Stage Decoupled Pipeline for TP >= 2 Ring All-Reduce MatMul:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): Ring Network Remote DMA Send/Recv in flight across chips.
  - Stage 3 (Tile k-2): VPU Vector Addition & Asynchronous VMEM -> HBM DMA start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index(axis_name)
  right_neighbor = lax.rem(my_id + 1, num_devices)

  num_rounds = num_devices - 1
  remote_send_sems = network_sems[0:num_rounds]
  remote_recv_sems = network_sems[num_rounds : 2 * num_rounds]

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
      j = lax.rem(tile_idx, num_n)
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      k_slice = pl.ds(k_idx * block_k, block_k)
      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sem,
        )
        copy_x.start()
      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sem,
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
          acc_vmem[acc_slot, :, :] = (
              acc_vmem[acc_slot, :, :] + sub_acc_k.astype(o_hbm_ref.dtype)
          )

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
              device_id={axis_name: right_neighbor},
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
            sem=hbm_out_sem,
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
    hbm_x_sem,
    hbm_w_sem,
    hbm_out_sem,
    *network_sems,
    num_devices: int,
    block_m: int,
    block_n: int,
    block_k: int,
    axis_name: str = "tp",
):
  """5-Stage Decoupled Pipeline for TP >= 2 All-to-All All-Reduce MatMul on Sunfish:

  - Stage 0 (Tile k+1): Asynchronous HBM -> VMEM DMA load of inputs X and W.
  - Stage 1 (Tile k):   MXU Matrix Multiplication on VMEM double buffer.
  - Stage 2 (Tile k-1): Simultaneous 1-to-(TP-1) point-to-point remote DMA in flight.
  - Stage 3 (Tile k-2): VPU multi-peer accumulation & Asynchronous VMEM -> HBM DMA start.
  - Stage 4 (Tile k-3): Asynchronous HBM Store completion wait.
  """
  m = x_hbm_ref.shape[0]
  n = w_hbm_ref.shape[1]
  num_m = m // block_m
  num_n = n // block_n
  num_tiles = num_m * num_n

  my_id = lax.axis_index(axis_name)
  num_peers = num_devices - 1
  remote_send_sems = network_sems[0:num_peers]
  remote_recv_sems = network_sems[num_peers : 2 * num_peers]

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
      j = lax.rem(tile_idx, num_n)
      return pl.ds(i * block_m, block_m), pl.ds(j * block_n, block_n)

    def load_hbm_inputs(tile_idx, slot, k_idx=0):
      m_slice, n_slice = get_tile_slices(tile_idx)
      k_slice = pl.ds(k_idx * block_k, block_k)
      with jax.named_scope("hbm_load_x"):
        copy_x = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[m_slice, k_slice],
            dst_ref=x_vmem.at[slot, :, :],
            sem=hbm_x_sem,
        )
        copy_x.start()
      with jax.named_scope("hbm_load_w"):
        copy_w = pltpu.make_async_copy(
            src_ref=w_hbm_ref.at[k_slice, n_slice],
            dst_ref=w_vmem.at[slot, :, :],
            sem=hbm_w_sem,
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
          acc_vmem[acc_slot, :, :] = (
              acc_vmem[acc_slot, :, :] + sub_acc_k.astype(o_hbm_ref.dtype)
          )

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
              device_id={axis_name: target_id},
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
            sem=hbm_out_sem,
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
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_0
      )
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
      cx2, cw2 = load_hbm_inputs(2, slot=0)
      compute_tile(1, in_slot=1, acc_slot=1, copy_x=cx1, copy_w=cw1)
      copy_1 = dispatch_network(acc_slot=1)
      store_0 = vpu_add_and_dispatch_hbm_store(
          0, acc_slot=0, out_slot=0, net_copy_descs=copy_0
      )
      compute_tile(2, in_slot=0, acc_slot=2, copy_x=cx2, copy_w=cw2)
      copy_2 = dispatch_network(acc_slot=2)
      wait_hbm_store(store_0)
      store_1 = vpu_add_and_dispatch_hbm_store(
          1, acc_slot=1, out_slot=1, net_copy_descs=copy_1
      )
      store_2 = vpu_add_and_dispatch_hbm_store(
          2, acc_slot=2, out_slot=0, net_copy_descs=copy_2
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
    x_scale: jax.Array | None = None,
    w_scale: jax.Array | None = None,
    mesh: jax.sharding.Mesh,
    axis_name: str = "tp",
    out_dtype: jnp.dtype = jnp.bfloat16,
    block_m: int = 1024,
    block_n: int = 1024,
    block_k: int = 4096,
    compute_m: int | None = None,
    compute_n: int | None = None,
    pipeline_mode: str = "5stage",
    interpret: Union[bool, pltpu.InterpretParams, None] = None,
) -> jax.Array:
  """Executes fused MatMul and All-Reduce across the provided mesh.

  Always utilizes a 5-stage decoupled software pipeline across all topologies:
  - TP=2: Direct 1-hop D2D/ICI exchange.
  - TP>=2 (All2All): 1-hop simultaneous point-to-point dispatches (Sunfish Boardfly).
  - TP>=2 (Ring): (TP-1) ring rounds with 2D Intra-D2D + ICI overlap (Ghostfish/Zebrafish).

  Args:
    x: Input tensor [M, K_local] (e.g. jnp.float8_e4m3fn or jnp.bfloat16).
    w: Weight tensor [K_local, N] (e.g. jnp.float8_e4m3fn or jnp.bfloat16).
    mesh: JAX device mesh containing the axis_name.
    axis_name: Name of the mesh axis representing Tensor Parallelism.
    out_dtype: Output and All-Reduce reduction dtype (default: jnp.bfloat16).
    block_m: Tiling block size along M dimension.
    block_n: Tiling block size along N dimension.
    block_k: Tiling block size along contracting K dimension.
    compute_m: Optional internal compute sub-tile size along M.
    compute_n: Optional internal compute sub-tile size along N.
    pipeline_mode: Collective scheduling mode ("5stage", "all2all", or "ring").
    interpret: Whether to run in Pallas interpret mode.

  Returns:
    Fully reduced output tensor [M, N] in out_dtype (BF16) replicated across shards.
  """
  num_devices = mesh.shape[axis_name]
  m = x.shape[0]
  k_local = x.shape[1] // num_devices
  n = w.shape[1]

  block_m = min(block_m, m)
  block_n = min(block_n, n)
  block_k = min(block_k, k_local)

  if interpret is None:
    interpret = (
        pltpu.InterpretParams() if jax.default_backend() == "cpu" else False
    )

  if num_devices == 2:
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(1,),
        scratch_shapes=[pltpu.SemaphoreType.DMA] * 5,
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_tp2,
        axis_name=axis_name,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

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
        scratch_shapes=[pltpu.SemaphoreType.DMA] * (3 + 2 * num_peers),
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_all2all,
        axis_name=axis_name,
        num_devices=num_devices,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

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
        scratch_shapes=[pltpu.SemaphoreType.DMA] * (3 + 2 * num_rounds),
    )

    kernel_fn = functools.partial(
        _fused_all_reduce_matmul_kernel_5stage_ring,
        axis_name=axis_name,
        num_devices=num_devices,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

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

  with jax.named_scope(f"fused_matmul_ar_m{block_m}_n{block_n}_k{block_k}"):
    out = shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(P(None, axis_name), P(axis_name, None)),
        out_specs=P(None, None),
        check_rep=False,
    )(x, w)

  if x_scale is not None and w_scale is not None:
    out = out * (x_scale * w_scale)
  elif x_scale is not None:
    out = out * x_scale
  elif w_scale is not None:
    out = out * w_scale
  return out
