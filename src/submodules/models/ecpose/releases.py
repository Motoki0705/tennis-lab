"""Source revision and release artifacts for the compared ECDet and ECPose models."""

from dataclasses import dataclass
from typing import Literal

SOURCE_URL = "https://github.com/Intellindust-AI-Lab/EdgeCrafter.git"
SOURCE_COMMIT = "d8752b75c9beaecfcdd0e5cd207bae3b6946ae23"
IMAGE_SIZE = (640, 640)  # height, width


@dataclass(frozen=True)
class EdgeCrafterRelease:
    family: Literal["ecdetseg", "ecpose"]
    config: str
    tag: str
    filename: str
    size: int
    sha256: str

    @property
    def has_pose(self) -> bool:
        return self.family == "ecpose"

    @property
    def download_url(self) -> str:
        return (
            "https://github.com/capsule2077/edgecrafter/releases/download/"
            f"{self.tag}/{self.filename}"
        )


RELEASES: dict[str, EdgeCrafterRelease] = {
    "ecdet-l-coco": EdgeCrafterRelease(
        "ecdetseg",
        "ecdet/ecdet_l.yml",
        "edgecrafterv1",
        "ecdet_l.pth",
        132351848,
        "60e48c51b33d70acf8e3a3faddbdb70b0876c64509b74cb623106cc7c31e85a1",
    ),
    "ecdet-x-coco": EdgeCrafterRelease(
        "ecdetseg",
        "ecdet/ecdet_x.yml",
        "edgecrafterv1",
        "ecdet_x.pth",
        200304616,
        "c87394be021b573fd8e25969b85a8670127120c9d0209ac7807fbadae1a5cec6",
    ),
    "ecdet-l-o365": EdgeCrafterRelease(
        "ecdetseg",
        "ecdet/ecdet_l.yml",
        "edgecrafterv1_o365",
        "ecdet_l_o3652coco.pth",
        132405122,
        "6259432a6184029e17cb008dcd3fa6ebf036c3e10cd56aa552a0d848ddcef19e",
    ),
    "ecdet-x-o365": EdgeCrafterRelease(
        "ecdetseg",
        "ecdet/ecdet_x.yml",
        "edgecrafterv1_o365",
        "ecdet_x_o3652coco.pth",
        200357826,
        "be9c77cd4def8aadd0cc679944ba57412f363de2d44f7c6b900289e61316314c",
    ),
    "ecpose-l-o365": EdgeCrafterRelease(
        "ecpose",
        "ecpose/ecpose_l_coco.yml",
        "edgecrafterv1_o365",
        "ecpose_l_o3652coco.pth",
        138759372,
        "091ad72c92ee5de3383176cbae8e8836e9cacf2b9fbec94c11c855fbfe35343c",
    ),
    "ecpose-x-o365": EdgeCrafterRelease(
        "ecpose",
        "ecpose/ecpose_x_coco.yml",
        "edgecrafterv1_o365",
        "ecpose_x_o3652coco.pth",
        206712140,
        "2d79c0fd21831e4f3e6d5ba7bfa10b5e6e8ecb2641180299d7a1bd968b33f4f6",
    ),
}
