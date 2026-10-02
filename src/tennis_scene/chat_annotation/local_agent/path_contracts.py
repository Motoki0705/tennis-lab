"""Explicit input/output scopes shared by every campaign command."""

from pathlib import Path

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
    RuntimePathRoots,
)

from .configuration import CampaignConfig, paths

COMMAND_PATHS = NonHydraPathBoundary(
    name='tennis_scene.chat_annotation.local_agent',
    fields=(
        BoundaryPathField('campaign', PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField('annotation_root', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField('attempt', PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, required=False),
        BoundaryPathField('manifest', PathRole.DATA, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, required=False),
        BoundaryPathField('annotation', PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True, required=False),
        BoundaryPathField('comparison', PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, required=False),
        BoundaryPathField('input_directory', PathRole.OUTPUT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, required=False),
        BoundaryPathField('output', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.ANY, required=False),
        BoundaryPathField('edits', PathRole.OUTPUT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, required=False),
    ),
)

INITIAL_PATHS = NonHydraPathBoundary(
    name='tennis_scene.chat_annotation.local_agent.init',
    fields=(
        BoundaryPathField('annotation_root', PathRole.DATA, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField('campaign', PathRole.OUTPUT, PathDirection.OUTPUT, PathKind.DIRECTORY, allow_role_root=True),
        BoundaryPathField('project', PathRole.PROJECT, PathDirection.INPUT, PathKind.DIRECTORY,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField('python', PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True),
        BoundaryPathField('checkpoint', PathRole.ARTIFACT, PathDirection.INPUT, PathKind.FILE,
                          must_exist=True, allow_role_root=True, required=False),
    ),
)


def campaign_resolver(config: CampaignConfig) -> PathResolver:
    return PathResolver(RuntimePathRoots(
        project_root=config.project_root,
        data_root=config.annotation_root,
        checkpoint_root=config.ball_checkpoint.parent if config.ball_checkpoint is not None else config.project_root / 'ckpt',
        artifact_root=config.campaign_dir,
        output_root=config.campaign_dir,
        cache_root=config.campaign_dir / 'cache',
        external_asset_root=config.codex_home,
    ))


def validate_command_paths(**arguments: Path | str) -> dict[str, Path]:
    """Caller-selected annotation files have their own read scope; outputs stay in the campaign."""
    config = paths()
    checked = COMMAND_PATHS.validate(
        {'campaign': config.campaign_dir, 'annotation_root': config.annotation_root, **arguments},
        resolver=campaign_resolver(config), independent_artifact_inputs=True,
    )
    return {name: checked.declared(name).path for name in arguments}


def validate_initial_paths(config: CampaignConfig) -> None:
    arguments = {'annotation_root': config.annotation_root, 'campaign': config.campaign_dir,
                 'project': config.project_root, 'python': config.python_executable}
    if config.ball_checkpoint is not None:
        arguments['checkpoint'] = config.ball_checkpoint
    # Validate the executable target without replacing the venv's launcher path.
    INITIAL_PATHS.validate(arguments, resolver=campaign_resolver(config), independent_artifact_inputs=True)


def output_name(value: str) -> str:
    """A sheet name is one filename component, not an alternate output root."""
    if not value or value != value.strip() or value in {'.', '..'} \
            or any(character in value for character in ('/', '\\', '\x00')):
        raise ValueError('image name must be one nonempty filename component')
    return value
