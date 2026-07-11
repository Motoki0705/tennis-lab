"""Deterministic layer (zorder) policy for 3D scene rendering.

mplot3d sorts whole artists by mean view depth by default; the huge ground
quads then hide flat artists (lines, rings, shadows) at many camera angles.
Scene renderers therefore disable depth sorting per axis
(:func:`enable_explicit_layering`) and assign explicit zorders following the
shared :class:`SceneLayer` convention, so every renderer produces the same
stable layer order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


class SceneLayer:
    """Shared zorder convention for 3D scene artists.

    ``surfaces < ground decals < net < structure < players < rings <
    trails < markers < ball < overlay``. Values match the defaults already
    hard-coded by :class:`~src.utils.rendering.court_renderer.CourtRenderer`
    (surface 0, lines 1, net 2, posts/band 3).
    """

    SURFACE = 0
    GROUND = 1  # court lines, contact shadows, ground movement trails
    NET = 2
    STRUCTURE = 3  # net posts, top band, center strap
    PLAYER = 4
    RING = 5  # bounce-impact rings
    TRAIL = 6  # airborne trajectory trails
    MARKER = 7  # direction arrows, annotations
    BALL = 10
    OVERLAY = 100  # HUD text


def enable_explicit_layering(ax: Axes3D) -> None:
    """Disable mplot3d depth sorting so explicit zorders take effect.

    The flag survives ``ax.clear()``, but frame renderers re-apply it anyway
    so a frame renders identically on a fresh or reused axis.
    """
    ax.computed_zorder = False
