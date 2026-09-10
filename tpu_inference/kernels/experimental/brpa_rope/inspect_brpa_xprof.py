#!/usr/bin/env python3
"""Inspect XProf execution events for bRPA in-kernel RoPE benchmark."""

import sys
from absl import app
from absl import flags
from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client

_FILTER = flags.DEFINE_string(
    "filter", "", "Filter session names containing this string"
)
_SESSION_ID = flags.DEFINE_string(
    "session_id", "", "Direct session ID to inspect"
)
_RUN = flags.DEFINE_string("run", "2", "Run to inspect ('1' or '2')")


def inspect_session(client, sid, title):
  print(
      f"\n=========================================================================="
  )
  print(f"SESSION: {title} ({sid})")
  print(
      f"=========================================================================="
  )
  hosts = client.get_hosts(sid, with_metadata=False)
  if not hosts:
    print(f"No hosts found for session {sid}")
    return
  host = hosts[0]
  xspace = client.get_xspace(sid, host, rpc_deadline_s=60, include_hlo="true")

  events = {}
  all_events_list = []
  for plane in xspace.planes:
    if plane.name.startswith("/host:"):
      continue
    event_metadata = {
        k: plane.event_metadata[k].name for k in plane.event_metadata
    }
    for line in plane.lines:
      for event in line.events:
        name = event_metadata.get(event.metadata_id, "")
        dur_us = event.duration_ps / 1e6
        start_us = event.offset_ps / 1e6
        all_events_list.append((start_us, dur_us, line.name, name))
        if dur_us > 0.1:
          if name not in events:
            events[name] = []
          events[name].append(dur_us)

  print("\nTop Operations by Total Duration:")
  for name, durs in sorted(
      events.items(), key=lambda x: sum(x[1]), reverse=True
  )[:25]:
    avg_d = sum(durs) / len(durs)
    print(
        f"  {name[:75]:<75} | Cnt: {len(durs):<4} | Avg: {avg_d:8.2f} µs |"
        f" Total: {sum(durs):8.2f} µs"
    )

  print("\nSearching for Copies, Transposes, Reshapes, and Layout Ops:")
  for name, durs in sorted(
      events.items(), key=lambda x: sum(x[1]), reverse=True
  ):
    name_lower = name.lower()
    if any(
        k in name_lower
        for k in [
            "copy",
            "transpose",
            "swapaxes",
            "reshape",
            "transpose",
            "dma",
            "permute",
        ]
    ):
      avg_d = sum(durs) / len(durs)
      print(
          f"  * {name:<75} | Cnt: {len(durs):<4} | Avg: {avg_d:8.2f} µs |"
          f" Total: {sum(durs):8.2f} µs"
      )

  # Check chronological sequence during first execution of pipeline
  all_events_list.sort(key=lambda x: x[0])
  print(
      "\nChronological Sequence of Device Events during first pipeline"
      " iteration:"
  )
  pipeline_starts = [e for e in all_events_list if "jit_test" in e[3]]
  if pipeline_starts:
    iter0_start = pipeline_starts[0][0]
    iter0_end = iter0_start + pipeline_starts[0][1] + 10.0
    for start_us, dur_us, line_name, name in all_events_list:
      if iter0_start <= start_us <= iter0_end:
        if (
            dur_us > 0.5
            and "Power" not in name
            and "CommonPjRt" not in name
            and "Throttle" not in name
        ):
          print(
              f"  [{start_us:12.2f} us] ({dur_us:8.2f} us) [{line_name:<15}]"
              f" {name}"
          )


def main(argv):
  del argv
  client = xprof_analysis_client.XprofAnalysisClient("fangfangz")

  if _SESSION_ID.value:
    inspect_session(client, _SESSION_ID.value, "Direct_Session")
    return

  sessions_run1 = {
      "0_Decoupled_Ref": "platforms-deepsea-8436557819037171049",
      "1_KV_Major_InKernelNormRoPE": "platforms-deepsea-8436557819037170711",
      "2_TokenMajor_FusedNorm_StridedDMA": (
          "platforms-deepsea-8436557819037171112"
      ),
      "3_KV_Major_StridedDMA_InKernelNorm": (
          "platforms-deepsea-8436557819037171513"
      ),
      "3b_TokenMajor_StridedDMA_InKernelNorm": (
          "platforms-deepsea-8436557819037167818"
      ),
      "4_KV_Major_2D_Pallas_Norm": "platforms-deepsea-8436557819037168219",
      "5_KV_Major_Fused_Pallas_Norm_RoPE": (
          "platforms-deepsea-8436557819037168620"
      ),
      "6_Head_Sharded_KV_2D_Pallas_Norm": (
          "platforms-deepsea-8436557819037169021"
      ),
      "7_Head_Sharded_KV_Fused_Norm_RoPE": (
          "platforms-deepsea-8436557819037169422"
      ),
  }

  sessions_run2 = {
      "0_Decoupled_Ref": "platforms-deepsea-309096689584021929",
      "1_KV_Major_InKernelNormRoPE": "platforms-deepsea-309096689584022743",
      "2_TokenMajor_FusedNorm_StridedDMA": (
          "platforms-deepsea-309096689584022056"
      ),
      "3_KV_Major_StridedDMA_InKernelNorm": (
          "platforms-deepsea-309096689584025465"
      ),
      "3b_TokenMajor_StridedDMA_InKernelNorm": (
          "platforms-deepsea-309096689584024778"
      ),
      "4_KV_Major_2D_Pallas_Norm": "platforms-deepsea-309096689584024091",
      "5_KV_Major_Fused_Pallas_Norm_RoPE": (
          "platforms-deepsea-309096689584023404"
      ),
      "6_Head_Sharded_KV_2D_Pallas_Norm": (
          "platforms-deepsea-309096689584022717"
      ),
      "7_Head_Sharded_KV_Fused_Norm_RoPE": (
          "platforms-deepsea-309096689584022030"
      ),
  }

  sessions = sessions_run2 if _RUN.value == "2" else sessions_run1
  for label, sid in sessions.items():
    if _FILTER.value and _FILTER.value not in label:
      continue
    try:
      inspect_session(client, sid, label)
    except Exception as e:
      print(f"Error on {label} ({sid}): {e}")
      import traceback

      traceback.print_exc()


if __name__ == "__main__":
  app.run(main)
