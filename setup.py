#!/usr/bin/env python

from setuptools import setup
from pathlib import Path

here = Path(__file__).parent

# Get the long description from the README file
with open(here / "README.md", encoding="utf-8") as f:
    long_description = f.read()

KEYWORDS = """\
analysis
data-analysis
gamma-spectroscopy
nuclear-physics
nuclear-spectrum-analysis
physics
python
spectroscopy
"""

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Environment :: Console
Intended Audience :: Information Technology
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Natural Language :: English
Operating System :: MacOS
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: UNIX
Programming Language :: C
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Information Analysis
Topic :: Scientific/Engineering :: Physics
"""

setup(
    name="hdtv",
    version="0.1.0",
    description="Bayesian deconvolution of nuclear spectra",
    url="https://github.com/op3/boris",
    author="Oliver Papst",
    maintainer_email="opapst@ikp.tu-darmstadt.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL",
    classifiers=CLASSIFIERS.strip().split("\n"),
    keywords=KEYWORDS.strip().replace("\n", " "),
    install_requires=["numpy", "pymc3", "uproot",],
    entry_points={
        "console_scripts": [
            "boris=boris.app:BorisApp",
            "sirob=boris.app:SirobApp",
        ]
    },
    packages=["boris",],
    tests_require=["pytest", "pytest-cov"],
)
