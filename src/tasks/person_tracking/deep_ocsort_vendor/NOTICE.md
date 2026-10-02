# Deep OC-SORT inference excerpts

Source: [GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT/tree/6bb51d027b137233f5c520b6fcc4f2ae387a6ba9),
`trackers/integrated_ocsort_embedding`, commit `6bb51d027b137233f5c520b6fcc4f2ae387a6ba9`. MIT notice in LICENSE.
Kalman core also credits Copyright 2014-2018 Roger R Labbe Jr. (MIT), retained here.
SORT precursor credits Alex Bewley, as in upstream source.

Only the inference Kalman box state/filter and IoU/direction/adaptive-weight functions are retained.
Unused dataset/encoder/GMC/grid/postprocessing/CLI and FilterPy convenience methods are omitted.
The filter receives column vectors only; its reshape_z call is the equivalent explicit column reshape.
The optional orig=True FilterPy branch is removed; the observation-centric filter is always used.
No extra packages, detector or encoder weights are loaded by this code.

The typed `deep_ocsort_pose.py` adapter owns frame/detection row validation, ID allocation, SciPy
assignment (one fixed backend), shared feature input, and a confidence-masked pose cost in both stages.
CMC/grid/new-KF are disabled as predeclared. Birth IDs are camera-instance-local, including interleaved
cameras. Virtual observations exist only inside the filter, never in the emitted detection rows.
This is Deep OC-SORT with a Lab-specific pose extension, not a reproduction of its benchmark scores.
