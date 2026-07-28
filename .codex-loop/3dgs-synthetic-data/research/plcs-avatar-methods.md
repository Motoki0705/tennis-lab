# PLCS Gaussian-avatar method research

Research checked 2026-07-28 UTC/JST. Exact official repository revisions are
machine-readable in `third_party/plcs_avatar/pins.json`. This document is the
single decision record; it avoids duplicating implementation instructions.

## Selection criteria

The PLCS generator needs a versioned human Gaussian asset controlled by the
repository's existing GVHMR SMPL-X output, deterministic single/multi-person
placement, complete labels, and one native NHT render with the court scene.
Standard 3DGS spherical-harmonic features are not NHT latent features, so an
upstream avatar may contribute geometry, opacity, scale, rotation, skinning,
and target renders, but its features must be fitted into NHT or trained there.

## Primary-paper and official-code matrix

| Method | Primary paper / official code / pinned commit | Applicability | Limits and decision |
|---|---|---|---|
| GaussianAvatar (CVPR 2024) | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_GaussianAvatar_Towards_Realistic_Human_Avatar_Modeling_from_a_Single_Video_CVPR_2024_paper.html), [official code](https://github.com/aipixel/GaussianAvatar), `d981c62238ef64e89dcc04719d2ebbb4758b080a` | The official implementation explicitly supports both SMPL and SMPL-X, stores fixed query LBS weights, and provides a novel-pose path. This most closely matches GVHMR's existing SMPL-X parameters. | Its pose-conditioned appearance network and standard differentiable Gaussian renderer cannot be copied into NHT. Selected as the primary control/asset-layout candidate under MIT; geometry and control will be exported, while appearance receives an explicit NHT fit. |
| HUGS: Human Gaussian Splats (CVPR 2024) | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kocabas_HUGS_Human_Gaussian_Splats_CVPR_2024_paper.html), [official Apple page](https://machinelearning.apple.com/research/hugs), [official code](https://github.com/apple/ml-hugs), `b65721a5946771053e4f1d0d68d06199bc1d8c07` | Directly models human Gaussians together with a static scene. The official code uses six nearest SMPL vertices, LBS-similarity confidence, and blended per-vertex transforms, making it the strongest comparison for scene composition. | SMPL rather than SMPL-X; standard SH appearance; Apple sample-code license requires review for redistribution. Selected as the comparative geometry-control candidate, not vendored. |
| GART: Gaussian Articulated Template Models (CVPR 2024 Highlight) | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Lei_GART_Gaussian_Articulated_Template_Models_CVPR_2024_paper.pdf), [project](https://www.jiahuilei.com/projects/gart/), [official code](https://github.com/JiahuiLei/GART), `16c11f8a5bb3ae249a9d04dc9d98c316e10f1126` | Learned forward skinning and latent bones can capture deformation outside the SMPL surface. MIT is integration-friendly. | Official README describes a pre-release, uses SMPL v1.1/CUDA 11.8, and the repository is small. Retained as fallback if fixed SMPL-X control fails the measured gate. |
| SplattingAvatar (CVPR 2024) | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Shao_SplattingAvatar_Realistic_Real-Time_Human_Avatars_with_Mesh-Embedded_Gaussian_Splatting_CVPR_2024_paper.html), [official code](https://github.com/initialneil/SplattingAvatar), `fec0ad3845f1d2e4ad4cdabd1b1c8c81cf10e41b` | Mesh-triangle embedding is a useful geometric reference because persistent barycentric attachments are deterministic. | The pinned official HEAD contains only `README.md`, despite setup claims, and is CC BY-NC-SA 4.0 with an older PyTorch/CUDA stack. Rejected as an executable candidate; its mesh-attachment idea is used only as the probe reference. |
| Animatable Gaussians (CVPR 2024) | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Animatable_Gaussians_Learning_Pose-dependent_Gaussian_Maps_for_High-fidelity_Human_Avatar_CVPR_2024_paper.html), [official project](https://animatable-gaussians.github.io/), [official code](https://github.com/lizhe00/AnimatableGaussians), `2b0f6e3b4c5af823414eb6d5f0b2e1a59954d114` | Front/back Gaussian maps and pose-dependent appearance target high-fidelity loose clothing. | Requires multi-view capture and a StyleUNet-like appearance system. Deferred until the lighter SMPL-X asset/control boundary is proven. |
| 3DGS-Avatar (CVPR 2024) | [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qian_3DGS-Avatar_Animatable_Avatars_via_Deformable_3D_Gaussian_Splatting_CVPR_2024_paper.html), [official code](https://github.com/mikeqzy/3dgs-avatar-release), `fc39bc5ec622d1de33f38b4b9c6d39b8485bdecd` | A non-rigid network can model residual cloth motion and provides a useful escalation path. | Python 3.7/PyTorch 1.12/CUDA 11.6, SMPL-only, and upstream 3DGS license inheritance raise integration cost. Deferred behind GART. |

## Current decision and failure policy

The first implementation trial compares GaussianAvatar-style fixed SMPL-X
query weights against HUGS-style six-neighbour transform blending on the local
licensed `SMPLX_NEUTRAL.npz`. Persistent barycentric mesh attachments are the
geometric reference. This isolates control quality before expensive appearance
training and prevents a good RGB network from hiding unstable geometry.

GVHMR SMPL-X is the authoritative control input. COCO17 keypoints are complete
labels and may later drive an explicit SMPL-X fitting/IK module, but the
underdetermined COCO17-to-SMPL-X inverse is never a silent fallback. A failed
or missing fit must reject the frame.

Neither selected method's appearance tensor is treated as NHT-compatible.
After geometry control converges, the avatar must be trained directly in NHT
or use the already established frozen-target feature-fit boundary with
calibrated multi-view renders. Single/multi-person composition will then apply
independent world transforms and stable instance IDs before one scene render.

## Representative trials and final P4 selection

The cycle-10 geometry screen exercised both viable candidates on the same nine
tennis-like SMPL-X frames and 512 persistent mesh attachments:

| Candidate | Mean / p95 / max attachment error | Result |
|---|---:|---|
| GaussianAvatar-style fixed query LBS | 0.876 / 2.560 / 5.451 mm | Passed; selected because it is marginally more accurate and the official implementation already supports SMPL-X. |
| HUGS-style six-neighbour transform blend | 0.893 / 2.624 / 5.478 mm | Passed; retained as a comparison/fallback, but adds a nearest-neighbour approximation without improving this motion set. |

The selected cycle-11 prototype samples 4,096 area-weighted anisotropic
Gaussians on the licensed neutral SMPL-X surface, interpolates all 55 joint
weights, and pushes each covariance through its blended linear transform. The
three canonical/ready/forehand frames are all emitted with no dropped joints or
frames. Maximum p95 mean-position error against persistent posed-mesh
attachments is 4.416 mm. An independent repeat reproduces the entire geometry
artifact byte for byte.

Appearance is initialized from zero and optimized only in the pinned NHT
feature space with frozen geometry, opacity, and deferred shader. It reaches
67.880 dB held-out masked PSNR, and the repeated fit reaches 67.922 dB. The
NHT/gsplat CUDA path is not byte deterministic: repeated latent features differ
by 0.01019 maximum and 0.000830 mean absolute value. This does not silently
become a byte-reproducibility claim. Across six native pose/view renders, the
measured effect is at most one uint8 LSB and 0.00221 mean LSB, with a 0.0414 dB
validation-PSNR delta. The explicit P4 tolerance is therefore feature
max/mean <=0.02/0.002, render max/mean <=1/0.01 LSB, and PSNR delta <=0.1 dB.

The generated green appearance is a mechanics prototype, not a captured human
identity or clothing-fidelity claim. P4 accepts the asset/control/NHT boundary;
P5 must compose independently transformed single/multi-person instances with
the court scene, stable identities, and complete visibility labels before
claiming dataset generation.
