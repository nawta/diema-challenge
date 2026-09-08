"""Model registry for string-based model lookup.

Provides a simple registry mapping model names (strings) to model classes.
All registered models must be subclasses of BaseModel.

See: diema_challenge_implementation_spec.md §6.4
Related: diema/models/__init__.py (auto-registration)

Ported from: an internal baseline with build_model() addition.
"""

from diema.models.base import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str, cls: type[BaseModel]) -> None:
    """Register a model class under the given name."""
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(f"Cannot register {cls}: must be a subclass of BaseModel")
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' is already registered (to {_REGISTRY[name]})")
    _REGISTRY[name] = cls


def get_model(name: str) -> type[BaseModel]:
    """Look up a registered model class by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


def build_model(config) -> BaseModel:
    """Build a model instance from config.

    Uses config.model.name (or config.model.type) to look up the class,
    then calls cls.from_config(config).

    Args:
        config: config namespace with model.name/type, skeleton, etc.

    Returns:
        Instantiated BaseModel subclass.
    """
    model_name = getattr(config.model, "name", None) or getattr(config.model, "type", None)
    if model_name is None:
        raise ValueError("Config must have model.name or model.type")
    cls = get_model(model_name)
    return cls.from_config(config)
