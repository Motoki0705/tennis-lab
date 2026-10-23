"""Byte pins for qualification of the published standard pipeline.

Acquisition and license decisions live in docs/installation/README.md and
LICENSES.md. Published detector/sample and SMPL pins retain their own manifests.
This catalog adds the third-party files needed for a complete installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES
from src.utils.model_weights import MODEL_REPO, PUBLISHED_WEIGHTS
from src.utils.sample_dataset import DATASET_REPO, PUBLISHED_FILES

PublicDocumentLanguage = Literal["en", "ja", "zh-CN"]


@dataclass(frozen=True)
class PublicDocumentLocale:
    suffix: str
    label: str


PUBLIC_DOCUMENT_LANGUAGES: dict[PublicDocumentLanguage, PublicDocumentLocale] = {
    "en": PublicDocumentLocale(".md", "English"),
    "ja": PublicDocumentLocale(".ja.md", "日本語"),
    "zh-CN": PublicDocumentLocale(".zh-CN.md", "简体中文"),
}


def public_document_language_from_filename(
    filename: str | Path,
) -> PublicDocumentLanguage:
    """Resolve a Markdown filename using the shared public-document suffixes.

    Bare ``.md`` filenames use English. Any additional suffix must be a supported
    language qualifier; unknown qualifiers never fall back to English.
    """
    suffix = "".join(Path(filename).suffixes)
    for language, locale in PUBLIC_DOCUMENT_LANGUAGES.items():
        if suffix == locale.suffix:
            return language
    raise ValueError(f"Unsupported public document suffix: {suffix!r}")


@dataclass(frozen=True)
class InstallationAsset:
    name: str
    filename: str  # CHECKPOINT-root relative
    sha256: str
    source: str
    license_en: str
    license_ja: str
    license_zh_cn: str
    local_acquisition: bool = True


_GVHMR_DOWNLOAD = (
    "https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD"
)
THIRD_PARTY_ASSETS = (
    InstallationAsset(
        "detector",
        "dino/checkpoint0029_4scale_swin.pth",
        "e61688afe3af91b25955e9f9601d04b42068327a2e076ee871828089aa4ffed5",
        "https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_",
        "DINO / Apache-2.0¹",
        "DINO / Apache-2.0¹",
        "DINO / Apache-2.0¹",
    ),
    InstallationAsset(
        "vitpose",
        "vitpose/vitpose-h-multi-coco.pth",
        "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc",
        _GVHMR_DOWNLOAD,
        "ViTPose / Apache-2.0¹",
        "ViTPose / Apache-2.0¹",
        "ViTPose / Apache-2.0¹",
    ),
    InstallationAsset(
        "hmr2",
        "hmr2/epoch=10-step=25000.ckpt",
        "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e",
        _GVHMR_DOWNLOAD,
        "HMR2 / MIT¹; separate SMPL terms apply",
        "HMR2 / MIT¹; SMPL条件は別",
        "HMR2 / MIT¹；SMPL 条款另行适用",
    ),
    InstallationAsset(
        "gvhmr",
        "gvhmr/gvhmr_siga24_release.ckpt",
        "4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91",
        _GVHMR_DOWNLOAD,
        "GVHMR non-commercial terms",
        "GVHMR非商用条件",
        "GVHMR 非商业用途条款",
    ),
    InstallationAsset(
        "body_model",
        "body_models/smplx/SMPLX_NEUTRAL.npz",
        "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992",
        "https://smpl-x.is.tue.mpg.de/",
        "SMPL-X Model License / registration and acceptance",
        "SMPL-X Model License / 登録・同意",
        "SMPL-X Model License / 注册并同意条款",
    ),
    InstallationAsset(
        "aflink",
        "person_tracking/AFLink_epoch20.pth",
        "b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4",
        "https://drive.google.com/file/d/1DFMUkL-dc-j8-fibcJIq-46Xoq_bFoO9/view",
        "AFLink weight terms unconfirmed",
        "AFLink重みの条件は未確認",
        "AFLink 权重条款尚未核实",
    ),
    InstallationAsset(
        "tracking_encoder",
        "player_association/person_vit_clip_reid.pth",
        "bedc8fcc37f296045df19d901b162dd032bfa85d9b4f2406eff8313536d444dc",
        "https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view",
        "CLIP-ReID / MIT¹",
        "CLIP-ReID / MIT¹",
        "CLIP-ReID / MIT¹",
    ),
    InstallationAsset(
        "smpl_faces",
        "body_models/smplh/neutral/model.npz",
        "a60a7e29d33f09ef1a6352907e2f485e161f471330ca40592764374063e751df",
        "https://mano.is.tue.mpg.de/",
        "MANO / SMPL+H License / registration and acceptance",
        "MANO / SMPL+H License / 登録・同意",
        "MANO / SMPL+H License / 注册并同意条款",
    ),
    InstallationAsset(
        "dinov3",
        "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
        "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/",
        "DINOv3 License / application required",
        "DINOv3 License / 申請",
        "DINOv3 License / 需提交申请",
    ),
)

_PUBLISHED_WEIGHT_LICENSES = {
    "court": ("DINOv3 License", "DINOv3 License", "DINOv3 License"),
    "ball": ("CC BY-NC 4.0", "CC BY-NC 4.0", "CC BY-NC 4.0"),
}

INSTALLATION_ASSETS = (
    *(
        InstallationAsset(
            weight.name,
            weight.filename,
            weight.sha256,
            f"https://huggingface.co/{MODEL_REPO}",
            *_PUBLISHED_WEIGHT_LICENSES[weight.name],
            local_acquisition=False,
        )
        for weight in PUBLISHED_WEIGHTS
    ),
    *THIRD_PARTY_ASSETS,
    *(
        InstallationAsset(
            f"smpl_{name}",
            str(source.checkpoint_path),
            source.sha256,
            source.source_url,
            "GVHMR / SMPL / SMPL-X terms",
            "GVHMR・SMPL・SMPL-X条件",
            "GVHMR / SMPL / SMPL-X 条款",
        )
        for name, source in SMPL_ASSET_SOURCES.items()
    ),
)


def _require_language(language: PublicDocumentLanguage) -> None:
    if language not in PUBLIC_DOCUMENT_LANGUAGES:
        raise ValueError(
            f"Unsupported asset table language: {language!r}; "
            f"expected one of {tuple(PUBLIC_DOCUMENT_LANGUAGES)}"
        )


def asset_table(language: PublicDocumentLanguage) -> str:
    """Render checkpoint destinations, sources and terms without exposing byte pins.

    Languages and filename suffixes follow ``PUBLIC_DOCUMENT_LANGUAGES``.
    The ¹ markers refer to the guide's note about upstream code licenses.
    """
    _require_language(language)
    header = {
        "en": "| Asset | Destination (relative to `ckpt/`) | Source | License / terms |",
        "ja": "| 資産 | 配置先（`ckpt/`相対） | 入手先 | ライセンス・条件 |",
        "zh-CN": "| 资产 | 保存位置（相对于 `ckpt/`） | 下载来源 | 许可证 / 条款 |",
    }[language]
    source_label = {"en": "Source", "ja": "配布元", "zh-CN": "下载来源"}[language]
    rows = [header, "| --- | --- | --- | --- |"]
    for asset in INSTALLATION_ASSETS:
        terms = {
            "en": asset.license_en,
            "ja": asset.license_ja,
            "zh-CN": asset.license_zh_cn,
        }[language]
        rows.append(
            f"| {asset.name} | `{asset.filename}` | "
            f"[{source_label}]({asset.source}) | {terms} |"
        )
    return "\n".join(rows)


def sample_asset_table(language: PublicDocumentLanguage) -> str:
    """Render the acquisition CLI's fixed sample file list without byte pins."""
    _require_language(language)
    header = {
        "en": "| Destination | Source / terms |",
        "ja": "| 配置先 | 入手先・条件 |",
        "zh-CN": "| 保存位置 | 下载来源 / 条款 |",
    }[language]
    source_label = {
        "en": "Published dataset",
        "ja": "公開Dataset",
        "zh-CN": "公开数据集",
    }[language]
    rows = [header, "| --- | --- |"]
    rows.extend(
        f"| `data/samples/{sample.filename}` | "
        f"[{source_label}](https://huggingface.co/datasets/{DATASET_REPO}) / CC BY-NC-ND 4.0 |"
        for sample in PUBLISHED_FILES
    )
    return "\n".join(rows)
