# Issue #695 B00 acceptance

The canonical video-to-report run completed all seven stages from NHT reconstruction through the three datasets and final report. Reconstruction exported 491 cameras and 217,355 scene points; alignment accepted both detected courts with disjoint fit and holdout evidence.

- Court accepted frames: 3077 from 3,112 proposals (98.875%), across 24 trajectory groups with zero split leakage and a 1.048605 m maximum adjacent arc step. All seven semantic classes have renderer-visible supervision in the canonical post-render semantic manifest. Repository CPU time was 87.896 s; 1,050.813 s of the 1,175.301 s wall time was inside the external NHT renderer boundary.
- BLCS source frames: 3072; planned, rendered, and labelled inventories are also 3,072. The default six-camera profile produced 18,432 samples in 70.649 s, using 485,025,257 bytes versus a 187,904,819,200-byte dense reference.
- PLCS aggregate global frames: 2234; both logical scenes independently preserve all 1,117 planned, rendered, and labelled frames. Each scene contains all three complete ACCAD sources (running, walking, and general) on the multi-object global timeline using per-frame SMPL-H Gaussian LBS on CUDA. Twelve total camera streams completed in 248.797 s, using 945,781,054 bytes versus a 296,475,033,600-byte dense reference. Cross-court source evaluation, camera tensors, tile binning, and device-to-host payloads are shared or batched.
- Camera authority is six cameras for `default` and two cameras for `broadcast`. PLCS assigns one complete logical scene to each accepted court (`court-000: 1`, `court-001: 1`), for a maximum count difference of zero.

The committed numbers come from the current canonical B00 workspace and were independently reopened by `test_b00_gpu_acceptance.py`; they are not smoke or fixture substitutions.
