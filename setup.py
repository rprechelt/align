from os import path
from setuptools import setup

# the align version
__version__ = "0.0.4"

# get the absolute path of this project
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# the standard setup info
setup(
    name="align",
    version=__version__,
    description="Fast and simple alignment of 1D signals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rprechelt/align",
    author="Remy L. Prechelt",
    author_email="prechelt@hawaii.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["signal", "processing", "alignment", "delay", "waveforms"],
    packages=["align"],
    python_requires=">=3.6*, <4",
    install_requires=["numpy", "scipy", "numba"],
    extras_require={
        "test": [
            "pytest",
            "black",
            "mypy",
            "isort",
            "coverage",
            "pytest-cov",
            "flake8",
        ],
    },
    scripts=[],
    project_urls={},
    package_data={"align": ["py.typed"]},
    include_package_data=True,
)
