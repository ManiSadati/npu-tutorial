# Local validation — 2026-09-09

- CANN 9.1.0 beta.3; target `dav-c310-vec`; simulator `Ascend950PR_9599`.
- Final `run.sh` passed with an empty environment except HOME and PATH,
  using the workspace's `toolchain` symlink (exit 0).
- Device compilation, host linking, three launches, comparisons and cleanup passed.
- Three FP32 vector-add trials: 256/256 exact results each.
- Successful run: `build/sim-r57ahypB/run.log`.
- All shell scripts passed `bash -n`; final device build has no warnings.

This validates the fixed 256-element CCE intrinsics example in the CANN A5
operator simulator, not on physical A5 hardware. The simulator driver stub
reports `Ascend950PR_9589`; the loaded configuration is `Ascend950pr_9599`.
