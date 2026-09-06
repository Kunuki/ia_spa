import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
VERSION = re.search(
    r'^__version__ = "([^"]+)"', (ROOT / "ia_spa" / "__init__.py").read_text(), re.M
).group(1)

# The placement algorithm itself only needs an array stack and a progress bar;
# ray tracing, plotting and the notebooks are optional extras.
CORE_REQUIREMENTS = [
    "numpy>=1.24",
    "scipy>=1.11",
    "tqdm>=4.66",
    "pyyaml>=6.0",
]

VIZ_REQUIREMENTS = [
    "matplotlib>=3.8",
    "plotly>=5.18",
    "kaleido>=0.2",       # plotly static image export
    "imageio>=2.33",
    "trimesh>=4.0",
    "scikit-image>=0.22",
    "pandas>=2.1",
]

setup(
    name="ia_spa",
    version=VERSION,
    description=(
        "Interference-Aware Submodular Placement Algorithm for optimal "
        "wireless transmitter placement"
    ),
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Lukas Taus, Richard Tsai, Jeffrey G. Andrews",
    url="https://github.com/Kunuki/ia_spa",
    license="MIT",
    packages=find_packages(include=["ia_spa", "ia_spa.*"]),
    python_requires=">=3.10",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "viz": VIZ_REQUIREMENTS,
        "dev": ["pytest>=7.0"],
        "all": VIZ_REQUIREMENTS + ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "ia-spa = ia_spa.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
