"""
This module defines the ProviderFactory, responsible for dynamically
instantiating API provider classes based on configuration.
"""
import importlib
import inspect
import pkgutil
from typing import Any, Dict, Type


class ProviderFactory:
    """
    A factory class for dynamically creating provider instances.

    This factory maintains a registry of provider classes and can instantiate
    them based on a given name and configuration. This allows for extensible
    provider support without modifying core application logic.
    """

    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[Any]) -> None:
        """
        Registers a provider class with the factory.

        Args:
            name: The name under which to register the provider (e.g., "google", "mistral").
            provider_class: The class object of the provider.
        """
        cls._registry[name] = provider_class

    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any]) -> Any:
        """
        Retrieves and instantiates a provider by its registered name.

        Args:
            name: The name of the provider to retrieve.
            config: The configuration dictionary to pass to the provider's constructor.

        Returns:
            An instance of the requested provider.

        Raises:
            ValueError: If the provider name is not registered.
        """
        provider_class = cls._registry.get(name)
        if not provider_class:
            raise ValueError(f"Provider '{name}' is not registered.")
        return provider_class(config)

    @classmethod
    def load_providers(cls, package_name: str) -> None:
        """
        Dynamically loads and registers all provider classes from a given package.

        This method iterates through all modules in the specified package, inspects
        their members, and registers any class that ends with "Provider" but does
        not start with "Base".

        Args:
            package_name: The name of the package to scan for providers (e.g., "src.providers").
        """
        package = importlib.import_module(package_name)
        for _, module_name, _ in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        ):
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and name.endswith("Provider")
                    and not name.startswith("Base")
                ):
                    provider_name = name.replace("Provider", "").lower()
                    cls.register_provider(provider_name, obj)

    @property
    def registry(self):
        return self._registry
