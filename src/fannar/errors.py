"""Hierarquia de exceções da fannar."""

from __future__ import annotations


class FannarError(Exception):
    """Classe base para todos os erros da fannar."""


class ValidationError(FannarError):
    """Erro de validação de entrada (shape, dtype, propriedade estrutural)."""


class NotPSDError(ValidationError):
    """A matriz não é semidefinida positiva dentro da tolerância exigida."""


class NotSymmetricError(ValidationError):
    """A matriz não é simétrica dentro da tolerância exigida."""


class ShapeError(ValidationError):
    """Shapes incompatíveis entre tensores."""


class DeviceMismatchError(FannarError):
    """Tensores em devices distintos onde se exige o mesmo device."""


class NumericalError(FannarError):
    """Falha numérica (ex.: decomposição não convergiu, divisão por ~0)."""


class ExtractionError(FannarError):
    """Falha na extração de representações de um modelo PyTorch."""


class ConfigurationError(FannarError):
    """Configuração inválida ou incompatível."""
