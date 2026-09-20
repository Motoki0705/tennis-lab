"""Prepare, resume, validate and train a reproducible appearance variant."""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.synthetic_data_generation.appearance.configuration import (
    AppearanceRuntimeConfig,
)
from src.synthetic_data_generation.appearance.generation import (
    next_request,
    record_result,
)
from src.synthetic_data_generation.appearance.workspace import load_manifest, prepare
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="run_appearance_variant",
    version_base="1.3",
    validation_boundary="synthetic.appearance_variant",
)
def main(config: DictConfig) -> None:
    runtime = AppearanceRuntimeConfig.from_config(config)
    action = runtime.action
    if action == "report":
        from src.synthetic_data_generation.appearance.reporting import (
            compare_training_runs,
        )

        output = compare_training_runs(
            runtime.report.roots,
            runtime.report.output_root,
        )
    elif action == "compare_models":
        from src.synthetic_data_generation.appearance.comparison import compare_models

        variant = runtime.variant
        output = compare_models(
            variant,
            runtime.comparison.output_root,
            runtime.comparison.reference,
            runtime.comparison.target_index,
            retry_failed_request=runtime.api_retry,
        )
    elif action == "derive":
        from src.synthetic_data_generation.appearance.derived import derive_variant

        output = derive_variant(
            runtime.derive.parent_root,
            runtime.variant.output_root,
            scene_id=runtime.variant.scene_id,
            sample_count=runtime.variant.sample_count,
            max_steps=runtime.variant.max_steps,
        )
    elif action == "prepare":
        variant = runtime.variant
        output = prepare(variant).model_dump(mode="json")
    else:
        root = runtime.variant.output_root.resolve()
        if action == "configure_api_key":
            from src.synthetic_data_generation.appearance.workspace import (
                configure_api_key_file,
            )

            assert runtime.variant.api is not None
            configure_api_key_file(root, runtime.variant.api.api_key_file)
            output = {
                "status": "configured",
                "api_key_file": str(runtime.variant.api.api_key_file),
            }
        elif action == "generate_batch":
            from src.synthetic_data_generation.appearance.batch import generate_batch

            output = generate_batch(
                root,
                concurrency=runtime.batch.concurrency,
                start_interval_seconds=runtime.batch.start_interval_seconds,
                indices=runtime.batch.indices,
                retry_failed_request=runtime.api_retry,
            )
        elif action in {"check_api", "generate"}:
            from src.synthetic_data_generation.appearance.openai_api import (
                check_api_setup,
                generate_next,
            )

            output = (
                check_api_setup(root)
                if action == "check_api"
                else generate_next(root, retry_failed_request=runtime.api_retry)
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
            assert runtime.result.accepted is not None
            assert runtime.result.path is not None
            assert runtime.result.request_id is not None
            output = record_result(
                root,
                runtime.result.request_id,
                runtime.result.path,
                accepted=runtime.result.accepted,
                review_notes=runtime.result.notes,
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
                output = enqueue_training(root, retry_failed_job=runtime.training_retry)
            else:
                output = execute_training(root)
        else:
            raise ValueError(f"Unknown appearance action: {action}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
