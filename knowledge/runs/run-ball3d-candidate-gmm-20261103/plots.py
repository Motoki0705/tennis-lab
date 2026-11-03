"""Experiment-only uncertainty figures; video/standard figures use foundation."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import chi2


def uncertainty_figure(report, path):
    figure = Figure(figsize=(12, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="nominal")
    for name, method in report["methods"].items():
        values = method["lovo"]["evaluation"]["all"]
        ax.plot([.5, .9, .95], [values["ellipse_coverage"][str(p)] for p in (.5, .9, .95)],
            marker="o", label=f"{name} (n={values['count']})")
    ax.set(xlabel="Nominal 2D Gaussian ellipse probability", ylabel="Fraction of scored LOVO references inside",
        title="Primary [505,1010): covariance coverage conditional on a scored prediction", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    figure.savefig(path, dpi=110)


def ellipsoid_figure(runs, path):
    figure = Figure(figsize=(13, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    # Predeclared representative source frames; no selection using GT error.
    for panel, frame in enumerate((100, 300, 600, 800)):
        ax = figure.add_subplot(1, 4, panel+1, projection="3d")
        for name, run in runs.items():
            result = run.result
            if result.covariance is None or not result.valid[frame]:
                continue
            theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 17), np.linspace(0, np.pi, 9))
            sphere = np.stack((np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)), -1)
            surface = result.positions[frame] + (sphere @ np.linalg.cholesky(result.covariance[frame]).T)*np.sqrt(chi2.ppf(.95, 3))
            ax.plot_wireframe(surface[..., 0], surface[..., 1], surface[..., 2], linewidth=.4, alpha=.7,
                label=name.removeprefix("candidate_gmm_"))
        ax.set(title=f"Frame {frame}: 95% 3D ellipsoids", xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax.legend(fontsize=6)
    figure.savefig(path, dpi=100)
