from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "ctensor",
        ["src/bindings.cpp", "src/tensor.cpp"],
        define_macros=[('VERSION_INFO', __version__)],
    ),
]

setup(
    name="llmgrad",
    version=__version__,
    author="user447-ui",
    description="A lightweight AI engine written in C++",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.6.0"],
    python_requires=">=3.7",
)
