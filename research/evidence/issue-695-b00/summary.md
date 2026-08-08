# Issue #695 B00 acceptance

The canonical video-to-report run completed all seven stages from NHT reconstruction through the three datasets and final report. Reconstruction exported 491 cameras and 217,204 scene points; alignment accepted both detected courts with disjoint fit and holdout evidence.

- Court accepted frames: 2996 from 3,048 proposals (98.294%), across 24 trajectory groups with zero split leakage and a 1.048581 m maximum adjacent arc step. All seven semantic classes have renderer-visible supervision. Repository CPU time was 74.164 s; 992.812 s of the 1,101.704 s wall time was inside the external NHT renderer boundary.
- BLCS source frames: 3072; planned, rendered, and labelled inventories are also 3,072. The default six-camera profile produced 18,432 samples in 75.799 s, using 484,821,899 bytes versus a 187,904,819,200-byte dense reference.
- PLCS global frames: 1117; planned, rendered, and labelled inventories are also 1,117. Three complete ACCAD sources (running, walking, and general) are composed on the global multi-object timeline using per-frame SMPL-H Gaussian LBS on CUDA. The six-camera dataset completed in 193.840 s, using 402,824,520 bytes versus a 148,237,516,800-byte dense reference.
- Camera authority is six cameras for `default` and two cameras for `broadcast`; observed multi-court assignment differences are at most one.

The committed numbers come from the current canonical B00 workspace and were independently reopened by `test_b00_gpu_acceptance.py`; they are not smoke or fixture substitutions.
