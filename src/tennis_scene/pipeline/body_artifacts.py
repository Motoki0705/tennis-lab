"""Require the current body inference lineage before publishing a saved scene."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from src.tennis_scene.pipeline.components.body_view_selection import (
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.fasthmr import FastHMRInput, FastHMROutput
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.recipes import (
    FASTHMR_RECIPE,
    STANDARD_RECIPE,
    RecipeName,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)


def require_current_body_artifacts(
    root: Path, scene_reference: Mapping[str, Any], *, recipe: RecipeName = STANDARD_RECIPE,
) -> dict[str, Any]:
    """Follow this scene's body dependencies, including saved feature integrity.

    Active artifacts elsewhere in the index cannot substitute for the actual
    dependencies of the scene being published. No models are constructed.
    """
    root = root.resolve()
    document = json.loads((root / 'scene.json').read_text())
    assert_current_component_lineage(document, root, {'scene_assembly': dict(scene_reference)})
    artifacts = document['artifacts']

    def read(node: str, reference: Any, schema: str, version: int) -> dict[str, Any]:
        if not isinstance(reference, dict) or (reference.get('schema'), reference.get('version')) != (schema, version):
            raise ValueError(f'Scene requires {node}: {schema} v{version}')
        if artifacts.get(node) != reference:
            raise ValueError(f'Scene body dependency is not the active {node}')
        return cast(dict[str, Any], read_component_descriptor(root, reference, node=node, source_sha256=document['source_sha256']))

    scene = read('scene_assembly', dict(scene_reference), 'scene_result', 2)
    placement_ref = scene['dependencies'].get('placement')
    placement = read('body_placement', placement_ref, 'placed_bodies', 1)
    recovered_ref = placement['dependencies'].get('recovered')
    if recipe == FASTHMR_RECIPE:
        recovered = read(FASTHMR_RECIPE, recovered_ref, 'body_parameters', 3)
        selection_ref = recovered['dependencies'].get('selection')
        selection = read('body_view_selection', selection_ref, 'body_view_selection', 1)
        body = ArtifactCodec(FastHMROutput).load(
            recovered['payload'], (root / recovered_ref['path']).parent, recovered['arrays'],
        )
        selected = ArtifactCodec(BodyViewSelectionOutput).load(
            selection['payload'], (root / selection_ref['path']).parent, selection['arrays'],
        )
        source = cast(ClipSource, restore_type(document['source'], ClipSource))
        body.validate_selection(FastHMRInput(source, selected))
        return {'body_placement': placement_ref, FASTHMR_RECIPE: recovered_ref, 'body_view_selection': selection_ref}
    recovered = read('gvhmr', recovered_ref, 'body_parameters', 2)
    features_ref = recovered['dependencies'].get('features')
    features = read('hmr2_features', features_ref, 'hmr2_features', 1)
    selection_ref = recovered['dependencies'].get('selection')
    selection = read('body_view_selection', selection_ref, 'body_view_selection', 1)
    if features['dependencies'].get('selection') != selection_ref:
        raise ValueError('GVHMR and HMR2 features refer to different body selections')
    # Decode only the selected inputs, checking array checksums and the
    # clip/person/segment/frame correspondence already required by GVHMR.
    images = ArtifactCodec(HMR2FeaturesOutput).load(features['payload'], (root / features_ref['path']).parent, features['arrays'])
    selected = ArtifactCodec(BodyViewSelectionOutput).load(selection['payload'], (root / selection_ref['path']).parent, selection['arrays'])
    source = cast(ClipSource, restore_type(document['source'], ClipSource))
    GVHMRInput(source, selected, images).validate()
    return {'body_placement': placement_ref, 'gvhmr': recovered_ref, 'hmr2_features': features_ref, 'body_view_selection': selection_ref}
