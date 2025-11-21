"""Factory for creating Tennis-DETR models with different versions."""

from __future__ import annotations

from torch import nn

from src.models.tennis_multi_cam_3d_pose.config import TennisDetrConfig
from src.models.tennis_multi_cam_3d_pose.config_v2 import TennisDetrV2Config
from src.models.tennis_multi_cam_3d_pose.config_v3 import TennisDetrV3Config
from src.models.tennis_multi_cam_3d_pose.model import TennisDETR
from src.models.tennis_multi_cam_3d_pose.model_v2 import TennisDETR_v2
from src.models.tennis_multi_cam_3d_pose.model_v2_5 import TennisDETR_v2_5
from src.models.tennis_multi_cam_3d_pose.model_v3 import TennisDETR_v3


def create_tennis_model(
    model_version: str = "v2",
    cfg: TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config | None = None,
) -> nn.Module:
    """指定されたバージョンのTennis-DETRモデルを生成する.

    Args:
        model_version (str): モデルバージョン ("v1", "v2", "v2_5", "v3")
            - v1: 元の単一エンコーダモデル（TennisDETR）
            - v2: 階層エンコーダ + 分離出力モデル（TennisDETR_v2）
            - v2_5: v2を拡張したカメラ/時間埋め込み付きモデル（TennisDETR_v2_5）
        cfg (TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config | None): 設定オブジェクト。Noneの場合はデフォルト設定を使用

    Returns:
        nn.Module: 指定されたバージョンのモデルインスタンス

    Raises:
        ValueError: サポートされていないモデルバージョンが指定された場合

    """
    if cfg is None:
        if model_version in {"v2", "v2_5"}:
            cfg = TennisDetrV2Config()
        elif model_version == "v3":
            cfg = TennisDetrV3Config()
        else:
            cfg = TennisDetrConfig()

    if model_version == "v3":
        if not isinstance(cfg, TennisDetrV3Config):
            raise ValueError("v3 model requires TennisDetrV3Config")
        return TennisDETR_v3(cfg)

    if model_version == "v2_5":
        if not isinstance(cfg, TennisDetrV2Config):
            raise ValueError("v2_5 model requires TennisDetrV2Config")
        return TennisDETR_v2_5(cfg)
    if model_version == "v2":
        if not isinstance(cfg, TennisDetrV2Config):
            raise ValueError("v2 model requires TennisDetrV2Config")
        return TennisDETR_v2(cfg)
    if model_version == "v1":
        if not isinstance(cfg, TennisDetrConfig):
            raise ValueError("v1 model requires TennisDetrConfig")
        return TennisDETR(cfg)

    available_versions = ["v1", "v2", "v2_5", "v3"]
    raise ValueError(
        f"Unsupported model version: {model_version}. "
        f"Available versions: {available_versions}"
    )


def get_available_model_versions() -> list[str]:
    """利用可能なモデルバージョンのリストを返す.

    Returns:
        list[str]: 利用可能なモデルバージョン名のリスト

    """
    return ["v1", "v2", "v2_5", "v3"]


def validate_config_for_version(
    cfg: TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config,
    model_version: str,
) -> None:
    """指定されたバージョンに対して設定が有効かを検証する.

    Args:
        cfg (TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config): 検証する設定オブジェクト
        model_version (str): モデルバージョン

    Returns:
        None: This function does not return a value.

    Raises:
        ValueError: 設定が無効な場合

    """
    if model_version in {"v2", "v2_5"}:
        if not isinstance(cfg, TennisDetrV2Config):
            raise ValueError(f"{model_version} model requires TennisDetrV2Config")
        # v2系では階層エンコーダパラメータが必要
        if cfg.intra_layers <= 0:
            raise ValueError("intra_layers must be positive for v2/v2_5 model")
        if cfg.inter_layers <= 0:
            raise ValueError("inter_layers must be positive for v2/v2_5 model")
        if cfg.temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive for v2/v2_5 model")

    elif model_version == "v3":
        if not isinstance(cfg, TennisDetrV3Config):
            raise ValueError("v3 model requires TennisDetrV3Config")
        if cfg.intra_layers <= 0:
            raise ValueError("intra_layers must be positive for v3 model")
        if cfg.inter_layers <= 0:
            raise ValueError("inter_layers must be positive for v3 model")
        if cfg.temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive for v3 model")

    elif model_version == "v1":
        if not isinstance(cfg, TennisDetrConfig):
            raise ValueError("v1 model requires TennisDetrConfig")
        # v1では単一エンコーダパラメータが必要
        if cfg.encoder_layers <= 0:
            raise ValueError("encoder_layers must be positive for v1 model")

    # 共通の検証
    if cfg.D_model <= 0:
        raise ValueError("D_model must be positive")
    if cfg.num_joints <= 0:
        raise ValueError("num_joints must be positive")
    if cfg.num_queries <= 0:
        raise ValueError("num_queries must be positive")


def create_default_config(
    model_version: str,
) -> TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config:
    """指定されたバージョンのデフォルト設定を生成する.

    Args:
        model_version (str): モデルバージョン

    Returns:
        TennisDetrConfig | TennisDetrV2Config | TennisDetrV3Config: デフォルト設定オブジェクト

    Raises:
        ValueError: サポートされていないモデルバージョンが指定された場合

    """
    if model_version in {"v2", "v2_5"}:
        return TennisDetrV2Config()
    if model_version == "v3":
        return TennisDetrV3Config()
    if model_version == "v1":
        return TennisDetrConfig()
    raise ValueError(f"Unknown model version: {model_version}")


if __name__ == "__main__":
    # テストコード
    print("Testing Tennis-DETR model factory...")

    # v1モデルのテスト
    print("\n1. Creating v1 model with default config:")
    model_v1 = create_tennis_model("v1")
    print(f"   Model type: {type(model_v1).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model_v1.parameters()):,}")

    # v2モデルのテスト
    print("\n2. Creating v2 model with default config:")
    model_v2 = create_tennis_model("v2")
    print(f"   Model type: {type(model_v2).__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model_v2.parameters()):,}")

    # 設定検証のテスト
    print("\n3. Testing config validation:")
    try:
        validate_config_for_version(create_default_config("v1"), "v1")
        print("   v1 config validation: PASSED")
    except ValueError as e:
        print(f"   v1 config validation: FAILED - {e}")

    try:
        validate_config_for_version(create_default_config("v2"), "v2")
        print("   v2 config validation: PASSED")
    except ValueError as e:
        print(f"   v2 config validation: FAILED - {e}")

    # 利用可能バージョンの表示
    print(f"\n4. Available model versions: {get_available_model_versions()}")
