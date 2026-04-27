from setuptools import setup, find_packages

setup(
    name="ia_spa",
    version="0.1.0",
    description="Interference-Aware Submodular Placement Algorithm for optimal wireless transmitter placement",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lukas Taus, Richard Tsai, Jeffrey G. Andrews",
    url="https://github.com/<your-handle>/ia-spa",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "matplotlib>=3.8",
        "plotly>=5.18",
        "trimesh>=4.0",
        "scikit-image>=0.22",
        "imageio>=2.33",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "pandas>=2.1",
        "kaleido>=0.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
