"""Service wiring for managed figures, private bytes and initial-only AI images."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import Event

from .editor import DocumentCompiler, GeminiEditor
from .gemini import Gemini
from .images import ImageGenerator, UnavailableImageGenerator, generate_image
from .jobs import BundleValidator, EditorFactory, PreparedEdit, preserve_resources
from .schema import (
    Candidate,
    CheckResult,
    ContractError,
    Document,
    EditRequest,
    Revision,
    RevisionInfo,
)
from .visuals import render_document, validate_visuals

CompilerFactory = Callable[[dict[str, bytes]], DocumentCompiler]


class ManagedResources:
    """A service-owned compiler and explicit unavailable browsing image capability.

    Image intent is represented by the writer's existing typed Figure contract,
    not guessed from words in the instruction. An unfulfilled generated image
    has kind=image, generated_concept provenance and an empty asset_ids list.
    The trusted compiler rejects it before semantic review or publication.
    """

    def __init__(self) -> None:
        self.images: ImageGenerator = UnavailableImageGenerator()

    def prepare(
        self,
        request: EditRequest,
        current: Revision,
        history: list[RevisionInfo],
        *,
        cancel: Event,
    ) -> PreparedEdit:
        prepared = preserve_resources(request, current, history, cancel=cancel)
        policy = self.images.capability()
        context = prepared.context.model_copy(
            update={
                "preferences": (
                    "Browsing image capability: " + policy.reason + "\n"
                    "If this instruction requires a new AI-generated raster image, "
                    "declare that unmet requirement as a Figure with kind=image, "
                    "provenance.kind=generated_concept, an empty asset_ids list and "
                    "an empty figure slot. Describe the intended image in its caption/alt. "
                    "Do not claim success with a text/SVG substitute, invented image bytes "
                    "or an existing image. Existing image relocation and caption edits "
                    "retain their asset IDs. SVG and managed graphs remain supported."
                )
            }
        )
        return PreparedEdit(context, prepared.assets)

    def _check_images(self, document: Document) -> None:
        for figure in document.figures:
            if (
                figure.kind == "image"
                and figure.provenance.kind == "generated_concept"
                and not figure.asset_ids
            ):
                if self.images.capability().available:
                    raise ContractError("image_generation_unavailable")
                generate_image(
                    figure.caption,
                    asset_id=figure.figure_id,
                    caption=figure.caption,
                    alt=figure.alt,
                    adapter=self.images,
                )
                raise ContractError("image_generation_unavailable")

    def compiler(
        self, assets: dict[str, bytes], *, transform: DocumentCompiler | None = None
    ) -> DocumentCompiler:
        def compile_document(document: Document) -> Document:
            self._check_images(document)
            if transform is not None:
                document = transform(document)
                if not isinstance(document, Document):
                    raise ContractError("invalid_document")
                self._check_images(document)
            return render_document(document)

        return compile_document

    def editor_factory(
        self, gemini: Gemini, *, compiler_factory: CompilerFactory | None = None
    ) -> EditorFactory:
        def create(prepared: PreparedEdit) -> GeminiEditor:
            transform = (
                compiler_factory(prepared.assets)
                if compiler_factory is not None
                else None
            )
            return GeminiEditor(
                gemini,
                compile_document=self.compiler(prepared.assets, transform=transform),
            )

        return create


def with_visual_validation(rendering: BundleValidator) -> BundleValidator:
    """Run real visual safety checks, then the separately supplied rendering check."""

    def validate(
        candidate: Candidate, assets: Mapping[str, bytes], *, cancel: Event
    ) -> CheckResult:
        if cancel.is_set():
            raise ContractError("cancelled")
        validate_visuals(candidate.document, dict(assets))
        if cancel.is_set():
            raise ContractError("cancelled")
        return rendering(candidate, assets, cancel=cancel)

    return validate
