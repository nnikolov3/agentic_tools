"""
This package initializes all API providers by importing them, which
triggers their registration with the ProviderFactory.
"""

from src.providers.provider_factory import ProviderFactory

ProviderFactory.load_providers(__name__)
