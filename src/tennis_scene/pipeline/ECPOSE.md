# ECPose person detection and pose

Select `recipe=ecpose` to use the published Objects365-to-COCO ECPose-L or ECPose-X model. The declaration for each camera is `ecpose → person_detection + person_pose_features`. The two readers publish the existing v2 artifact types from the same inference and source rows. Appearance features, tracking, selection and body recovery consume those types through their usual inputs.

Place the pinned EdgeCrafter repository under `paths.external_asset_root` and the chosen release checkpoint under `paths.checkpoint_root`. The selected variant, checkpoint checksum, source, batch size, confidence and box margin participate in the model identity. Plan and publication identities check the release checkpoint even for reuse without inference. PyTorch restores the model on CPU before moving it to the configured device. See the [release pins and license](../../../knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md).

```bash
# Register this CUDA command with the shared training queue.
.venv/bin/python -m src.tennis_scene.scripts.run_pipeline \
  recipe=ecpose \
  '+ecpose={repository:EdgeCrafter,checkpoint:ecpose/ecpose_l_o3652coco.pth,model:ecpose-l-o365,backend:pt32,batch_size:1,confidence:0.3,box_margin:0.10}' \
  'video_paths=[match/cam0.mp4,match/cam1.mp4,match/cam2.mp4,match/cam3.mp4]' \
  'camera_ids=[cam0,cam1,cam2,cam3]'
```

`pt32` accepts an explicit CPU or CUDA device. `pt16` runs with CUDA FP16 autocast. `trt16` requires CUDA and an explicit `ecpose.engine` directory under the checkpoint root. Use an engine for the same release and a profile covering batch one through the configured batch size. Use the [EdgeCrafter conversion commands](../../../knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md) to create and qualify the engine. The camera stream is read in batches of up to `ecpose.batch_size`, including a smaller final batch. Model ownership is shared across cameras through the pipeline runtime.

The official head supplies one score per person and COCO17 coordinates. The confidence option retains scores strictly above the cutoff. Boxes come from the envelope of in-image COCO17 points plus the configured margin per side. The third joint value records coordinate availability as 1 or 0, and a missing joint is `[0,0,0]`. Set the score cutoff and box margin explicitly. The example uses 0.30 and 0.10. The accuracy benchmark's cutoff is documented separately.

The recipe requires person observations enabled and duplicate box merging disabled. Direct annotation is available through the standard recipe. The component store, clip publication and output file for ECPose use `recipes/ecpose`. Reuse requires artifacts produced by the declared components with the same inputs, model, backend and decode settings. `recipe=standard` is the default.

Dataset publication and completed-annotation reuse check each camera's ECPose result and its two views against the declared execution identity. The scene must reference both views from the same ECPose result.
