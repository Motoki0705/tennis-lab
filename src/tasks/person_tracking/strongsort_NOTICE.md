# StrongSORT++ paper-based inference implementation

Du, Zhao, Song, Zhao, Su, Gong, Meng: [StrongSORT: Make DeepSORT Great Again](https://arxiv.org/html/2202.13514v2),
arXiv:2202.13514v2 / IEEE TMM 2023, sections III–V, equations 2–15 and figure 3.
The paper is CC BY 4.0. The implementation was written for this repository from
the mathematical description, not copied from the GPL-3.0 reference code.
Official numerical defaults were checked at `dyhBUPT/StrongSORT` commit
`ee995076da5083e28d0da1f885297df62705ebd7`. Checkpoint key/shapes and synthetic
reference outputs were used for interoperability checks. This is not a formal
legal clean-room certification.

The comparison substitutes the fixed COCO .30 detections and precomputed CLIP
for YOLOX/BoT, disables ECC on the fixed cameras, and retains detection-row
identity and real-observation masks. GSI output remains a separate reconstruction;
it is never relabelled as a real detection. No pose cost is added to StrongSORT.
This is a specified adaptation, not a reproduction of the MOT benchmark scores.
All comparison settings have one source, the [run-9 addendum](../../../knowledge/runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md).

AFLink checkpoint: the official README's [Google Drive folder](https://drive.google.com/drive/folders/1Zk6TaSJPbpnqbz1w4kfhkKFCEzQbjfp_),
file ID `1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9`, `AFLink_epoch20.pth`, 4,348,705 bytes,
SHA-256 `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4`.
The reference repository is GPL-3.0. Separate checkpoint licence terms could not
be found in its README or the distribution folder. The weight is not distributed
here or claimed to be MIT; it is used only for the local research comparison.
Reimplementation does not resolve weight licensing. See the provisional
[issue decision](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903293645)
before any production adoption or redistribution.
