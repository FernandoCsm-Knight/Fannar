"""Visualização (matplotlib; plotly opcional). Não calcula geometria."""

from .heatmaps import plot_distance_heatmap, plot_gram_heatmap
from .metric_energy import (
    crop_patch,
    image_to_hwc,
    plot_image_energy_panel,
    plot_orthogonal_energy_panel,
    plot_patch_montage,
    save_current_figure,
)
from .plots import plot_spectrum
from .distance_diagnostics import plot_sentence_distance_diagnostics
from .token_trajectories import plot_token_trajectories
from .projection import plot_rkhs_spectral_projection_2d, plot_spectral_projection
from .radar import plot_explanatory_radar
from .scatter import plot_semantic_scatter_3d, semantic_overlap_score

__all__ = [
    "plot_spectral_projection",
    "plot_rkhs_spectral_projection_2d",
    "plot_sentence_distance_diagnostics",
    "plot_token_trajectories",
    "plot_image_energy_panel",
    "plot_orthogonal_energy_panel",
    "plot_patch_montage",
    "crop_patch",
    "image_to_hwc",
    "save_current_figure",
    "plot_explanatory_radar",
    "plot_gram_heatmap",
    "plot_distance_heatmap",
    "plot_spectrum",
    "plot_semantic_scatter_3d",
    "semantic_overlap_score",
]
