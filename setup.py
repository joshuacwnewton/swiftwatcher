"""Setup script for swiftwatcher"""

import os.path
from setuptools import setup

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.rst")) as fid:
    README = fid.read()

# This call to setup() does all the work
setup(
    name="swiftwatcher",
    version="0.1.0",
    description="Count chimney swifts in video files",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://github.com/joshuacwnewton/swiftwatcher",
    author="Joshua Newton",
    author_email="joshuacwnewton@gmail.com",
    license="GNU GPL-3.0",
    classifiers=[
        "License :: OSI Approved :: GNU GPL-3.0 License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    packages=["swiftwatcher"],
    include_package_data=True,
    install_requires=[
        "numpy", "opencv-python", "pandas", "scikit-image", "scipy"
    ],
    entry_points={"console_scripts":
                  ["swiftwatcher = swiftwatcher.__main__:main"]},
)