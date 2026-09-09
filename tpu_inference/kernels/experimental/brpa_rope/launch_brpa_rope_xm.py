r"""XManager Launcher for Batched Ragged Paged Attention (bRPA) with In-Kernel RoPE.

Usage on Ghostfish (GF):
  /google/bin/releases/xmanager/cli/xmanager.par launch \
    experimental/users/fangfangz/kernels/brpa_rope/launch_brpa_rope_xm.py -- \
    --platform=gf=1x1x1 \
    --bq_sz=256 \
    --bkv_sz=256 \
    --bq_c_sz=256 \
    --xm_resource_alloc=group:msca-dynamic/tpu-perf-team-dynamic-xm
"""

import getpass
import time
from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

_PLATFORM = flags.DEFINE_string(
    'platform',
    'gf=1x1x1',
    'TPU Accelerator specification (e.g. sfs=1x1x1, gf=1x1x1, gfc=1x1x1).',
)

_PRIORITY = flags.DEFINE_integer('priority', 200, 'Borg task priority.')

_Q_LEN = flags.DEFINE_integer('q_len', 4096, 'Query token sequence length.')
_KV_LEN = flags.DEFINE_integer('kv_len', 4096, 'Key/Value token cache length.')
_NUM_Q_HEADS = flags.DEFINE_integer('num_q_heads', 64, 'Number of query heads.')
_NUM_KV_HEADS = flags.DEFINE_integer('num_kv_heads', 8, 'Number of KV heads.')
_HEAD_DIM = flags.DEFINE_integer('head_dim', 128, 'Head dimension.')
_PAGE_SIZE = flags.DEFINE_integer('page_size', 128, 'Page size in tokens.')
_NUM_REQUESTS = flags.DEFINE_integer(
    'num_requests', 1, 'Number of concurrent requests.'
)

_BQ_SZ = flags.DEFINE_integer('bq_sz', 256, 'Prefill query block size.')
_BKV_SZ = flags.DEFINE_integer('bkv_sz', 256, 'Prefill KV block size.')
_BQ_C_SZ = flags.DEFINE_integer(
    'bq_c_sz', 256, 'Prefill query compute tile size.'
)

_NUM_WARMUP = flags.DEFINE_integer(
    'num_warmup', 5, 'Number of warmup iterations.'
)
_NUM_ITERS = flags.DEFINE_integer(
    'num_iters', 20, 'Number of benchmark iterations.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if '=' in _PLATFORM.value:
    platform_code, topology = _PLATFORM.value.split('=', 1)
  else:
    platform_code, topology = 'gf', _PLATFORM.value

  platform_code = platform_code.lower()
  is_ghostfish = platform_code in ('gf', 'gfc', 'ghostfish')
  resource_code = 'gf' if is_ghostfish else 'sfs'
  deepsea_ver = 'ghostfish' if is_ghostfish else 'sunfish'

  req_kwargs = {
      'priority': _PRIORITY.value,
      resource_code: topology,
  }
  if not is_ghostfish:
    req_kwargs['cpu'] = 4
    req_kwargs['ram'] = 16 * 1024**3
    req_kwargs['architecture'] = xm.Architecture.ARM

  requirements = xm.JobRequirements(**req_kwargs)

  extra_args = [
      f'--q_len={_Q_LEN.value}',
      f'--kv_len={_KV_LEN.value}',
      f'--num_q_heads={_NUM_Q_HEADS.value}',
      f'--num_kv_heads={_NUM_KV_HEADS.value}',
      f'--head_dim={_HEAD_DIM.value}',
      f'--page_size={_PAGE_SIZE.value}',
      f'--num_requests={_NUM_REQUESTS.value}',
      f'--bq_sz={_BQ_SZ.value}',
      f'--bkv_sz={_BKV_SZ.value}',
      f'--bq_c_sz={_BQ_C_SZ.value}',
      f'--num_warmup={_NUM_WARMUP.value}',
      f'--num_iters={_NUM_ITERS.value}',
      f'--deepsea_version={deepsea_ver}',
      '--alsologtostderr',
  ]

  env_vars = {
      'LIBTPU_INIT_ARGS': '--xla_tpu_use_dynamic_smem_negotiation=true',
  }

  platform_title = 'GF' if is_ghostfish else 'SF'
  with xm_abc.create_experiment(
      experiment_title=(
          f'bRPA In-Kernel RoPE [{platform_title}]'
          f' [q={_Q_LEN.value},kv={_KV_LEN.value},bq={_BQ_SZ.value}]'
      ),
  ) as experiment:

    [executable] = experiment.package([
        xm.bazel_binary(
            label='//experimental/users/fangfangz/kernels/brpa_rope:benchmark_brpa_rope',
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=xm_abc.bazel_args.for_requirements(requirements)
            + ('--dynamic_mode=off',),
        ),
    ])

    job = xm.Job(
        executable=executable,
        executor=xm_abc.Borg(
            requirements=requirements,
            borg_user=getpass.getuser(),
            logs_read_access_roles=['all'],
            use_auto_host_resources=is_ghostfish,
        ),
        args=extra_args,
        env_vars=env_vars,
    )

    experiment.add(job)


if __name__ == '__main__':
  app.run(main)
