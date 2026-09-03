"""Tests for Fused All-Reduce MatMul Pallas Kernel supporting FP8 (f8e4m3fn) & BF16."""

import os

if "XLA_FLAGS" not in os.environ:
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.collectives.fused_all_reduce_matmul import fused_all_reduce_matmul

P = jax.sharding.PartitionSpec


class FusedAllReduceMatmulTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.num_available_devices = jax.device_count()

  def _reference_all_reduce_matmul(self, x, w, mesh, axis_name="tp"):
    """Reference implementation computing MatMul in FP32 accumulation and All-Reduce in BF16."""

    def _shard_fn(x_shard, w_shard):
      local_prod = jnp.matmul(
          x_shard, w_shard, preferred_element_type=jnp.float32
      ).astype(jnp.bfloat16)
      return lax.psum(local_prod, axis_name=axis_name)

    return shard_map(
        _shard_fn,
        mesh=mesh,
        in_specs=(P(None, axis_name), P(axis_name, None)),
        out_specs=P(None, None),
        check_rep=False,
    )(x, w)

  @parameterized.named_parameters(
      (
          "prefill_fp8_attention_out_tp2_smoke",
          256,
          256,
          512,
          2,
          jnp.float8_e4m3fn,
          "5stage",
      ),
      (
          "prefill_fp8_attention_out_tp2",
          512,
          512,
          1024,
          2,
          jnp.float8_e4m3fn,
          "5stage",
      ),
      (
          "prefill_bf16_attention_out_tp2",
          512,
          512,
          1024,
          2,
          jnp.bfloat16,
          "5stage",
      ),
  )
  def test_prefill_tp2(self, m, k_local, n, num_devices, in_dtype, pipeline_mode):
    if self.num_available_devices < num_devices:
      self.skipTest(
          f"Requires at least {num_devices} devices, got"
          f" {self.num_available_devices}"
      )

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ["tp"])

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    x_f32 = (
        jax.random.normal(k1, (m, k_local * num_devices), dtype=jnp.float32)
        * 0.05
    )
    w_f32 = (
        jax.random.normal(k2, (k_local * num_devices, n), dtype=jnp.float32)
        * 0.05
    )

    x = x_f32.astype(in_dtype)
    w = w_f32.astype(in_dtype)

    ref_out = self._reference_all_reduce_matmul(x, w, mesh=mesh, axis_name="tp")
    fused_out = fused_all_reduce_matmul(
        x,
        w,
        mesh=mesh,
        axis_name="tp",
        out_dtype=jnp.bfloat16,
        block_m=min(m, 256),
        block_n=min(n, 512),
        block_k=min(k_local, 256),
        pipeline_mode=pipeline_mode,
    )

    self.assertEqual(fused_out.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(fused_out, dtype=np.float32),
        np.array(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=1.0,
    )

  @parameterized.named_parameters(
      (
          "decode_fp8_tp8_all2all",
          128,
          256,
          512,
          8,
          jnp.float8_e4m3fn,
          "all2all",
      ),
      (
          "decode_fp8_tp8_ring",
          128,
          256,
          512,
          8,
          jnp.float8_e4m3fn,
          "ring",
      ),
      (
          "decode_bf16_tp8_all2all",
          128,
          256,
          512,
          8,
          jnp.bfloat16,
          "all2all",
      ),
      (
          "decode_bf16_tp8_ring",
          128,
          256,
          512,
          8,
          jnp.bfloat16,
          "ring",
      ),
  )
  def test_decode_tp8(self, m, k_local, n, num_devices, in_dtype, pipeline_mode):
    if self.num_available_devices < num_devices:
      self.skipTest(
          f"Requires at least {num_devices} devices, got"
          f" {self.num_available_devices}"
      )

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ["tp"])

    key = jax.random.PRNGKey(123)
    k1, k2 = jax.random.split(key)

    x_f32 = (
        jax.random.normal(k1, (m, k_local * num_devices), dtype=jnp.float32)
        * 0.05
    )
    w_f32 = (
        jax.random.normal(k2, (k_local * num_devices, n), dtype=jnp.float32)
        * 0.05
    )

    x = x_f32.astype(in_dtype)
    w = w_f32.astype(in_dtype)

    ref_out = self._reference_all_reduce_matmul(x, w, mesh=mesh, axis_name="tp")
    fused_out = fused_all_reduce_matmul(
        x,
        w,
        mesh=mesh,
        axis_name="tp",
        out_dtype=jnp.bfloat16,
        block_m=min(m, 128),
        block_n=min(n, 256),
        block_k=min(k_local, 128),
        pipeline_mode=pipeline_mode,
    )

    self.assertEqual(fused_out.dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(fused_out, dtype=np.float32),
        np.array(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=1.0,
    )

  def test_strided_dma_swiglu_pipeline_tp2(self):
    """Verifies that Strided DMA correctly consumes SwiGLU FP8 outputs and matches sharded baseline."""
    num_devices = 2
    if self.num_available_devices < num_devices:
      self.skipTest(
          f"Requires at least {num_devices} devices, got"
          f" {self.num_available_devices}"
      )

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ["tp"])

    m = 256
    k_local = 512
    k_total = k_local * num_devices  # 1024
    n = 512

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    gate = jax.random.normal(k1, (m, k_total), dtype=jnp.float32) * 0.05
    up = jax.random.normal(k2, (m, k_total), dtype=jnp.float32) * 0.05
    w = (jax.random.normal(k3, (k_total, n), dtype=jnp.float32) * 0.05).astype(
        jnp.float8_e4m3fn
    )

    # 1. SwiGLU activation + per-token dynamic FP8 quantization inside shard_map
    def _swiglu_and_quantize(g, u):
      def _shard_fn(g_shard, u_shard):
        act_shard = jax.nn.silu(g_shard) * u_shard
        amax_local = jnp.max(jnp.abs(act_shard), axis=-1, keepdims=True)
        amax_global = lax.psum(amax_local, axis_name="tp")
        scale = 448.0 / jnp.maximum(amax_global, 1e-12)
        act_fp8_shard = jnp.clip(act_shard * scale, -448.0, 448.0).astype(
            jnp.float8_e4m3fn
        )
        return act_fp8_shard

      return shard_map(
          _shard_fn,
          mesh=mesh,
          in_specs=(P(None, "tp"), P(None, "tp")),
          out_specs=P(None, "tp"),
          check_rep=False,
      )(g, u)

    act_fp8 = _swiglu_and_quantize(gate, up)

    # Reference Down-Projection
    ref_out = self._reference_all_reduce_matmul(
        act_fp8, w, mesh=mesh, axis_name="tp"
    )

    # Fused Kernel with Qwen3-32B sharded inputs (Zero All-Gather, Zero frontend slice)
    fused_out = fused_all_reduce_matmul(
        act_fp8,
        w,
        mesh=mesh,
        axis_name="tp",
        out_dtype=jnp.bfloat16,
        block_m=min(m, 256),
        block_n=min(n, 512),
        block_k=min(k_local, 256),
        pipeline_mode="5stage",
    )

    # Assert numerical accuracy against reference
    np.testing.assert_allclose(
        np.array(fused_out, dtype=np.float32),
        np.array(ref_out, dtype=np.float32),
        rtol=2e-2,
        atol=1.0,
    )

  def test_zero_all_gather_hlo(self):
    """Verifies at the HLO compiler IR level that no all-gather ops are generated for pre-sharded weights/activations."""
    num_devices = 2
    if self.num_available_devices < num_devices:
      self.skipTest(
          f"Requires at least {num_devices} devices, got"
          f" {self.num_available_devices}"
      )

    devices = jax.devices()[:num_devices]
    mesh = jax.sharding.Mesh(devices, ["tp"])

    m = 256
    k_local = 512
    k_total = k_local * num_devices
    n = 512

    sharding_x = jax.sharding.NamedSharding(mesh, P(None, "tp"))
    sharding_w = jax.sharding.NamedSharding(mesh, P("tp", None))

    def pipeline(x, w):
      return fused_all_reduce_matmul(
          x,
          w,
          mesh=mesh,
          axis_name="tp",
          out_dtype=jnp.bfloat16,
          block_m=256,
          block_n=256,
          block_k=256,
          pipeline_mode="5stage",
      )

    x_dummy = jax.device_put(
        jnp.zeros((m, k_total), dtype=jnp.float8_e4m3fn), sharding_x
    )
    w_dummy = jax.device_put(
        jnp.zeros((k_total, n), dtype=jnp.float8_e4m3fn), sharding_w
    )

    hlo = jax.jit(pipeline).lower(x_dummy, w_dummy).as_text("hlo")

    # Assert no all-gather ops in HLO
    self.assertNotIn("all-gather", hlo.lower())
    self.assertNotIn("all_gather", hlo.lower())


if __name__ == "__main__":
  absltest.main()
