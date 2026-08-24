# SPDX-License-Identifier: Apache-2.0

import os

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference import utils
from tpu_inference.kernels.collectives import fused_all_reduce_matmul

jax.config.parse_flags_with_absl()

P = jax.sharding.PartitionSpec


@jtu.with_config(jax_numpy_dtype_promotion='standard')
class FusedAllReduceMatmulTest(jtu.JaxTestCase):

    @parameterized.product(
        m=[16, 128, 1024],
        pipeline_mode=["5stage", "ring", "all2all"],
    )
    def test_fused_all_reduce_matmul(self, m, pipeline_mode):
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("Need at least 2 devices for All-Reduce test")

        axis_name = "tp"
        mesh = utils.make_optimized_mesh((num_devices,), (axis_name,))
        k_local = 2560 // num_devices
        n = 5120

        # Adjust k_local and n if needed to be multiples of 128
        k_local = max(128, (k_local // 128) * 128)
        n = max(128, (n // 128) * 128)

        prng_key = jax.random.key(42)
        k0, k1 = jax.random.split(prng_key, 2)
        x = jax.random.normal(k0, (m, k_local), dtype=jnp.bfloat16)
        w = jax.random.normal(k1, (k_local, n), dtype=jnp.bfloat16)

        sharded_x = jax.device_put(
            x, jax.sharding.NamedSharding(mesh, P(None, axis_name))
        )
        sharded_w = jax.device_put(
            w, jax.sharding.NamedSharding(mesh, P(axis_name, None))
        )

        output = fused_all_reduce_matmul(
            sharded_x,
            sharded_w,
            mesh=mesh,
            axis_name=axis_name,
            out_dtype=jnp.bfloat16,
            pipeline_mode=pipeline_mode,
        )

        @jax.jit
        def expected_fn(x_s, w_s):
            def _shard_fn(x_i, w_i):
                return jax.lax.psum(x_i @ w_i, axis_name=axis_name)

            return jax.shard_map(
                _shard_fn,
                mesh=mesh,
                in_specs=(P(None, axis_name), P(axis_name, None)),
                out_specs=P(None, None),
            )(x_s, w_s)

        expected = expected_fn(sharded_x, sharded_w)
        self.assertAllClose(output, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
