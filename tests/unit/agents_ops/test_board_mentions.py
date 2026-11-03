"""Board mention syntax without registry, SQLite, runtime or notifier fixtures."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest


@pytest.fixture
def mentions(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], list[str]]:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    messages = importlib.import_module("board.messages")
    # Assignment/liveness is deliberately absent: it is downstream of parsing.
    monkeypatch.setattr(
        messages, "load_names", lambda: ["Aaron", "Adam", "Adrian", "Amina"]
    )
    return cast(Callable[[str], list[str]], messages._mentions)


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize(
    "template",
    [
        "> {fence}text\n> {mentions}\n> {fence}",
        "> > {fence}text\n> > {mentions}\n> > {fence}",
        ">>>{fence}text\n>>>{mentions}\n>>>{fence}",
        "   >   > {fence}text\n > > {mentions}\n> > {fence}",
        "- {fence}text\n  {mentions}\n  {fence}",
        "* {fence}text\n  {mentions}\n  {fence}",
        "+ {fence}text\n  {mentions}\n  {fence}",
        "1. {fence}text\n   {mentions}\n   {fence}",
        "12) {fence}text\n    {mentions}\n    {fence}",
        "123456789. {fence}text\n           {mentions}\n           {fence}",
        "- Example\n\n  {fence}text\n  {mentions}\n  {fence}",
        "12. Example\n\n    {fence}text\n    {mentions}\n    {fence}",
        "-    Example\n\n     {fence}text\n     {mentions}\n     {fence}",
        "- Example\n  1. {fence}text\n     {mentions}\n     {fence}",
        "> - {fence}text\n>   {mentions}\n>   {fence}",
        "- > {fence}text\n  > {mentions}\n  > {fence}",
        "-\t{fence}text\n\t{mentions}\n\t{fence}",
        ">\t{fence}text\n>\t{mentions}\n>\t{fence}",
        "> Example\n1. {fence}text\n   {mentions}\n   {fence}",
        "> Example\n- {fence}text\n  {mentions}\n  {fence}",
        "> Example\n{fence}text\n{mentions}\n{fence}",
        "- Example\n{fence}text\n{mentions}\n{fence}",
        "1. Example\n2. {fence}text\n   {mentions}\n   {fence}",
        "1) Example\n0) {fence}text\n   {mentions}\n   {fence}",
        "1. Example\n2.\n   Example\n3. {fence}text\n   {mentions}\n   {fence}",
        "> 1. Example\n> 2. {fence}text\n>    {mentions}\n>    {fence}",
        "- 1. Example\n  2. {fence}text\n     {mentions}\n     {fence}",
        "1. > Example\n2. {fence}text\n   {mentions}\n   {fence}",
    ],
)
def test_container_fences_exclude_mentions_and_preserve_adjacent_prose(
    mentions: Callable[[str], list[str]], template: str, fence: str
) -> None:
    body = template.format(
        fence=fence, mentions="@aLl @Aaron @Adam @Amina @NotInTheCatalog @human"
    )
    assert mentions(body) == []
    assert mentions(body + "\n\n宛先は@Adrianです。") == ["adrian"]


@pytest.mark.parametrize("fence", ["```", "~~~"])
@pytest.mark.parametrize(
    "template",
    [
        "> {fence}\n> {mentions}\n@{other}",
        "> > {fence}\n> > {mentions}\n> @{other}",
        "- {fence}\n  {mentions}\n@{other}",
        "- {fence}\n  {mentions}\n- @{other}",
        "1. {fence}\n   {mentions}\n2. @{other}",
        "> - {fence}\n>   {mentions}\n> - @{other}",
        "- > {fence}\n  > {mentions}\n  @{other}",
        "- Example\n  - {fence}\n    {mentions}\n  @{other}",
        "> Example\n1. {fence}\n   {mentions}\n@{other}",
        "> - Example\n1. {fence}\n   {mentions}\n@{other}",
        "1. Example\n2. {fence}\n   {mentions}\n@{other}",
    ],
)
def test_unclosed_fence_ends_at_container_boundary(
    mentions: Callable[[str], list[str]], template: str, fence: str
) -> None:
    body = template.format(fence=fence, mentions="@all @Aaron", other="Adrian")
    assert mentions(body) == ["adrian"]


@pytest.mark.parametrize(
    "template",
    [
        "> Example\n2. ~~~\n   @{other}",
        "> Example\n2. ```\n   @{other}",
        "> Example\n0) ~~~\n   @{other}",
        "> Example\n02. ~~~\n    @{other}",
        "> > Example\n2. ~~~\n   @{other}",
        "> - Example\n2. ~~~\n   @{other}",
        "> 1. Example\n2. ~~~\n   @{other}",
        "- Example\n2. ~~~\n   @{other}",
        "* Example\n2. ```\n   @{other}",
        "1. Example\n2) ~~~\n   @{other}",
        "1) Example\n0. ~~~\n   @{other}",
        "- > Example\n  2. ~~~\n     @{other}",
        "- 1. Example\n2. ~~~\n   @{other}",
        "> - 1. Example\n> 2. ~~~\n>    @{other}",
        "> Example\n*\n2. ~~~\n   @{other}",
        "> Example\n+\n2. ~~~\n   @{other}",
        "> Example\n1.\n2. ~~~\n   @{other}",
        "+ Example\n*\n2. ~~~\n   @{other}",
        "- Example\n1.\n2. ~~~\n   @{other}",
        "> Example\n~~\n@{other}",
        "- Example\n~~\n@{other}",
        "> Example\n```invalid`info\n@{other}",
        "- Example\n```invalid`info\n@{other}",
        "> Example\n    ~~~\n    @{other}",
        "- Example\n  2. ~~~\n     @{other}",
    ],
)
def test_lazy_container_continuation_keeps_prose_mentions(
    mentions: Callable[[str], list[str]], template: str
) -> None:
    assert mentions(template.format(other="Adrian")) == ["adrian"]
