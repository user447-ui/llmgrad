from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

__version__ = "0.2.0"
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

ext_modules = [
    Pybind11Extension(
        "ctensor",
        ["src/bindings.cpp", "src/tensor.cpp"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="llmgrad",
    version=__version__,
    author="user447-ui",
    description="A lightweight C++ deep learning engine with Python bindings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.10.0"],
    python_requires=">=3.8",
)
