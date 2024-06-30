import platform
import sys
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="whisper/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []
if sys.platform.startswith("linux") and platform.machine() == "x86_64":
    requirements.append("triton>=2.0.0,<3")

setup(
    name="whisper-onnx",
    py_modules=["whisper"],
    version=read_version(),
    description="Robust Speech Recognition via Large-Scale Weak Supervision",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="rhysdg",
    url="https://github.com/rhysdg",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    package_data={
        '': ['*.npz', '*.txt', '*.json', '*.onnx'],
    },
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ],
    entry_points={
        "console_scripts": ["whisper=whisper.transcribe:cli"],
    },
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
