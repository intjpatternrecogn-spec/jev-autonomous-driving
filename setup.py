"""Compatibility installer for offline environments with older setuptools."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="jev-autonomous-driving",
    version="0.1.0",
    description="JEV-inspired candidate-embedding decisions for driving research",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["jev_driving", "jev_driving.*"]),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["jev-driving=jev_driving.cli:main"]},
)
