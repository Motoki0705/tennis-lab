"""Prepare, resume, validate and train a reproducible appearance variant."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.appearance.contracts import VariantConfig
from src.synthetic_data_generation.appearance.generation import (
    next_request,
    record_result,
)
from src.synthetic_data_generation.appearance.workspace import load_manifest, prepare
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs", config_name="run_appearance_variant", version_base="1.3"
)
def main(config: DictConfig) -> None:
    action = str(config.action)
    if action == "report":
        from src.synthetic_data_generation.appearance.reporting import (
            compare_training_runs,
        )

        output = compare_training_runs(
            [Path(str(root)) for root in config.report.roots],
            Path(str(config.report.output_root)),
        )
    elif action == "compare_models":
        from src.synthetic_data_generation.appearance.comparison import compare_models

        variant = VariantConfig.model_validate(
            OmegaConf.to_container(config.variant, resolve=True)
        )
        output = compare_models(
            variant,
            Path(str(config.comparison.output_root)),
            Path(str(config.comparison.reference)),
            int(config.comparison.target_index),
            retry_failed_request=bool(config.api_retry),
        )
    elif action == "derive":
        from src.synthetic_data_generation.appearance.derived import derive_variant

        output = derive_variant(
            Path(str(config.derive.parent_root)),
            Path(str(config.variant.output_root)),
            scene_id=str(config.variant.scene_id),
            sample_count=int(config.variant.sample_count),
            max_steps=int(config.variant.max_steps),
        )
    elif action == "prepare":
        variant = VariantConfig.model_validate(
            OmegaConf.to_container(config.variant, resolve=True)
        )
        output = prepare(variant).model_dump(mode="json")
    else:
        root = Path(str(config.variant.output_root)).resolve()
        if action == "configure_api_key":
            from src.synthetic_data_generation.appearance.workspace import (
                configure_api_key_file,
            )

            configure_api_key_file(root, Path(str(config.variant.api.api_key_file)))
            output = {
                "status": "configured",
                "api_key_file": str(config.variant.api.api_key_file),
            }
        elif action == "generate_batch":
            from src.synthetic_data_generation.appearance.batch import generate_batch

            output = generate_batch(
                root,
                concurrency=int(config.batch.concurrency),
                start_interval_seconds=float(config.batch.start_interval_seconds),
                indices=list(config.batch.indices)
                if config.batch.indices is not None
                else None,
                retry_failed_request=bool(config.api_retry),
            )
        elif action in {"check_api", "generate"}:
            from src.synthetic_data_generation.appearance.openai_api import (
                check_api_setup,
                generate_next,
            )

            output = (
                check_api_setup(root)
                if action == "check_api"
                else generate_next(root, retry_failed_request=bool(config.api_retry))
            )
        elif action == "next_request":
            request = next_request(root)
            output = (
                {
                    "request": request.model_dump(mode="json"),
                    "execution_arguments": request.tool_arguments()
                    if request.provider == "builtin_imagegen"
                    else {
                        **(request.api_parameters or {}),
                        "prompt": request.prompt,
                        "image_paths": request.referenced_image_paths,
                    },
                }
                if request
                else {"request": None, "status": load_manifest(root).status}
            )
        elif action == "record_result":
            if config.result.accepted is None or config.result.path is None:
                raise ValueError(
                    "result.path and explicit result.accepted are required"
                )
            output = record_result(
                root,
                str(config.result.request_id),
                Path(str(config.result.path)),
                accepted=bool(config.result.accepted),
                review_notes=str(config.result.notes),
            ).model_dump(mode="json")
        elif action in {"finalize", "train", "execute_training"}:
            from src.synthetic_data_generation.appearance.nht import (
                enqueue_training,
                execute_training,
                finalize,
            )

            if action == "finalize":
                output = finalize(root)
            elif action == "train":
                output = enqueue_training(
                    root, retry_failed_job=bool(config.training_retry)
                )
            else:
                output = execute_training(root)
        else:
            raise ValueError(f"Unknown appearance action: {action}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
