"""Relatórios textuais (markdown) a partir de geometrias."""

from __future__ import annotations

from ..types import LayerGeometry


def layer_report_markdown(geo: LayerGeometry, name: str = "camada") -> str:
    """Gera um resumo em markdown dos principais diagnósticos da camada."""
    lines = [f"# Relatório geométrico — {name}", ""]
    C = geo.K_total.shape[0]
    lines.append(f"- Objetos (C): {C}")
    cov = geo.coverage.detail
    lines.append(f"- Cobertura espectral: r={cov['r']}/{cov['C']} "
                 f"(τ={cov['tau']}, fração capturada={cov['captured_fraction']:.3f})")
    if geo.residual is not None and geo.residual.value is not None:
        lines.append(f"- Energia residual (gradiente): {float(geo.residual.value):.3f}")
    fid2 = float(geo.spectrum.eigenvalues.clamp_min(0)[:2].sum()
                 / geo.spectrum.eigenvalues.clamp_min(0).sum().clamp_min(1e-12))
    lines.append(f"- Fidelidade 2D (traço): {fid2:.3f}")
    if geo.profile is not None:
        lines.append("")
        lines.append("## Perfil explicativo")
        for ax, val in geo.profile.as_dict().items():
            shown = f"{val:.3f}" if val is not None else "n/d"
            lines.append(f"- {ax}: {shown}")
    return "\n".join(lines)
