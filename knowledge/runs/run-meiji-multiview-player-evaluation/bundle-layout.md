# Meiji 4-camera player evaluation bundle

These six clips come from `meiji_4cam/multiview-v1`. Run `sha256sum -c SHA256SUMS` from this directory. Pin the SHA-256 of `manifest.json` in each comparison. Paths inside this bundle are relative to this directory. Source video, scene and component paths are relative to the pinned source benchmark.

## Cohort and quality audit

- `manifest.json` lists the source and generator pins, ordered clip, camera and player IDs, and original frame indices with an actual GVHMR result for at least one player.
- `quality_counts.csv` counts boxes and joints per clip, camera and player on the fixed frame cohort. Reason codes report the first applicable exclusion. Root rejection codes come from SceneResult and describe root placement. An invalid SMPL mesh can coexist with a valid root.
- `projection_audit.json` reports residuals against scene ViTPose and triangulation, including their sample counts, and pseudo-box agreement with the projected SMPL mesh at multiple margins. It includes the fixed-frame transformation, face and left/right checks.
- `vitpose_smoke.json` exercises matching and scoring on stored ViTPose and its expanded keypoint envelopes. Placement and visibility use some of the same observations, which couples audit results to the original reconstruction.
- `clips/<clip>/camN.mp4` shows native-size source images at the saved inference indices. The label gives the original frame number and the video plays at half the source FPS. Green is projected SMPL COCO17, magenta is scene ViTPose, and amber is the pseudo-box. Thin labelled boxes show a geometrically available box with an exclusion reason. The representative PNG and its original frame number are pinned in the manifest.

## Array axes

`P` is the clip-local player ID axis, `C` the camera axis, `T` the full source timeline, and `J` the COCO17 order in the manifest. Every clip NPZ stores the axis IDs and evaluation frame indices. Coordinates are metres in the Z-up court frame or native top-left pixel x/y as named.

| File and arrays | Shape and meaning |
|---|---|
| `pseudo_gt.npz / joints_court_m` | `(P,T,J,3)` from valid placed SMPL at actual GVHMR frames. Missing data is NaN. |
| `keypoints_px, cheirality, in_image, depth_m` | `(P,C,T,J,2)` pixels and `(P,C,T,J)` depth and projection masks. Behind-camera pixels are NaN. |
| `boxes_xyxy, mesh_boxes_xyxy, box_reasons` | `(P,C,T,4)` rectangle arrays and `(P,C,T)` scoring status. Unavailable boxes are NaN. Box bounds use `[0,width] × [0,height]`. |
| `gvhmr_inferred, scene_smpl_valid, root_rejection_code` | `(P,T)` exact recovered frame, archived mesh validity and archived root reason. |
| `vitpose_keypoints_xyc, same_view_observed, joint_reasons, likely_inter_player_occlusion` | `(P,C,T,J,3)` stored observations and `(P,C,T,J)` evidence and pose scoring masks. Status 0 marks an eligible joint. |
| `person_support_views, joint_support_views` | `(P,T)`, `(P,T,J)` counts of supporting camera observations. |
| `observation_error_px, triangulation_error_px, triangulation_error_m` | `(P,C,T,J)` projection distances and `(P,T,J)` court-space distances where diagnostics exist. |

## Fixed body-recovery inputs

`body_recovery_manifest.json` gives each clip's checksummed request NPZ, media, adopted component descriptors and underlying array SHA-256 values. Use the source benchmark to open GVHMR camera-relative parameters and the corresponding scene placement. The pinned crop and pose arrays are available in `clips/<clip>/body_recovery_inputs.npz`.

| Arrays | Shape and meaning |
|---|---|
| `selected_view_index, selected_intrinsic` | `(P,)` index into `camera_ids` and `(P,3,3)` selected camera K. |
| `segment_index, boxes_xys, keypoints_xyc` | `(P,T)`, `(P,T,3)`, `(P,T,J,3)` with the exact saved segment number, square crop center and side in pixels, and pose request. Segment -1 means no request. Outside requests crops and poses are NaN. |
| `selected_view_observed` | `(P,T)` flag for a saved request with actual detection support. Within requests a false value denotes an interpolated crop and missing pose input. Joint confidence indicates joint availability in the original request. |
| `K, R, t` | `(C,3,3)`, `(C,3,3)`, `(C,3)` pinhole model, with camera coordinates `R @ X_court + t`. |
| `source_frame, clip_time_sec, video_time_sec` | `(T,)` clip-relative frame number, nominal timestamp since clip start and timestamp on the synced video timeline. |
| `triangulated_joints_m, triangulated_valid, triangulated_rejection_code` | `(P,T,J,3)`, `(P,T,J)`, `(P,T,J)` preserved skeleton and masks. |
| `triangulated_inliers, triangulated_reprojection_px` | `(P,C,T,J)` support and residuals from the input skeleton. |
| `player_frame_ambiguous, player_id_ambiguous` | `(P,T)` tracking ambiguity flags from the input component. |
