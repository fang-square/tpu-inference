#!/usr/bin/env python3
"""Inspect XProf execution events for bRPA in-kernel RoPE benchmark."""

import sys
from absl import app
from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client


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
  for name, durs in sorted(events.items(), key=lambda x: sum(x[1]), reverse=True):
    name_lower = name.lower()
    if any(k in name_lower for k in ["copy", "transpose", "swapaxes", "reshape", "transpose", "dma", "permute"]):
      avg_d = sum(durs) / len(durs)
      print(
          f"  * {name:<75} | Cnt: {len(durs):<4} | Avg: {avg_d:8.2f} µs | Total: {sum(durs):8.2f} µs"
      )

  # Check chronological sequence around RPAm
  all_events_list.sort(key=lambda x: x[0])
  print("\nChronological Sequence of Device Events (first 30 significant events):")
  shown = 0
  for start_us, dur_us, line_name, name in all_events_list:
    if dur_us > 1.0 and not "Power" in name and not "CommonPjRt" in name:
      print(f"  [{start_us:12.2f} us] ({dur_us:8.2f} us) [{line_name}] {name}")
      shown += 1
      if shown >= 35:
        break


def main(argv):
  del argv
  client = xprof_analysis_client.XprofAnalysisClient("fangfangz")
  sessions = {
      "0_Decoupled_Ref": "platforms-deepsea-16584335618764035432",
      "1_KV_Major_InKernelNormRoPE": "platforms-deepsea-16584335618764037346",
      "2_TokenMajor_FusedNorm_StridedDMA": (
          "platforms-deepsea-16584335618764034845"
      ),
      "3_KV_Major_StridedDMA_InKernelNorm": (
          "platforms-deepsea-16584335618764036440"
      ),
      "3b_TokenMajor_StridedDMA_InKernelNorm": (
          "platforms-deepsea-16584335618764038035"
      ),
      "4_KV_Major_2D_Pallas_Norm": "platforms-deepsea-1181391949813709243",
  }
  for label, sid in sessions.items():
    try:
      inspect_session(client, sid, label)
    except Exception as e:
      print(f"Error on {label} ({sid}): {e}")
      import traceback
      traceback.print_exc()


if __name__ == "__main__":
  app.run(main)
