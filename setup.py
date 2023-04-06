from setuptools import setup, find_packages

requirements = [
    "numpy",
    "torch",
    "scipy",
]

setup(
    name="spingflow",
    version="0.1.0",
    author="Olivier Malenfant-Thuot",
    author_email="malenfantthuotolivier@gmail.com",
    description="GflowNets exploration on the Ising model",
    url="https://github.com/OMalenfantThuot/gflownet",
    packages=find_packages(),
    install_requires=requirements,
)
