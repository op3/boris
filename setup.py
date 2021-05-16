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
Development Status :: 4 - Beta
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
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Information Analysis
Topic :: Scientific/Engineering :: Physics
"""

setup(
    name="boris",
    version="0.3.0",
    description="Bayesian deconvolution of nuclear spectra",
    url="https://github.com/op3/boris",
    author="Oliver Papst",
    maintainer_email="opapst@ikp.tu-darmstadt.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL",
    classifiers=CLASSIFIERS.strip().split("\n"),
    keywords=KEYWORDS.strip().replace("\n", " "),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pymc3",
        "uproot3",
        "uproot",
    ],
    entry_points={
        "console_scripts": [
            "boris=boris.boris_app:BorisApp",
            "boris2spec=boris.boris2spec_app:Boris2SpecApp",
            "makematrix=boris.makematrix_app:MakeMatrixApp",
            "sirob=boris.sirob_app:SirobApp",
        ]
    },
    packages=[
        "boris",
    ],
    tests_require=["pytest", "pytest-cov"],
)
