# Local validation — 2026-09-09

- CANN 9.1.0 beta.3; target `dav-c310-vec`; simulator `Ascend950PR_9599`.
- PTO-ISA checkout: `896d8ec69aaf5b623fead5afcae7a657fa784a2b`.
- Kernel compilation, host link, simulator execution and cleanup succeeded.
- All three FP32 vector-add trials passed: 256/256 exact results each.
- Successful run: `build/sim-DhflR0NW/run.log` (exit 0).
- Final build also passed with an empty environment except HOME and PATH,
  using the workspace's `toolchain` symlink.
- All shell scripts passed `bash -n`.

This validates the fixed 256-element vector-add example through CANN's A5
operator simulator. It does not establish arbitrary-kernel support or physical
A5 hardware correctness/performance. The simulator's driver stub reports
`Ascend950PR_9589`; the loaded simulator configuration is `Ascend950pr_9599`.
