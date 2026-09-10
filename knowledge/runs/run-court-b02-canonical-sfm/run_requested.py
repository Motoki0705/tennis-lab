from pathlib import Path
import sys
from omegaconf import OmegaConf
from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.pipeline.application import build_scene_pipeline_runner
path=Path(sys.argv[1]);cfg=OmegaConf.load(path);runtime=ScenePipelineConfiguration.from_config(cfg)
runner=build_scene_pipeline_runner(runtime,resolved_config_yaml=OmegaConf.to_yaml(cfg,resolve=True,sort_keys=True))
manifest=runner.run(runtime.request)
print('COMPLETE',runtime.request.scene_id,manifest.stages[runtime.request.through_stage].status.value,flush=True)
