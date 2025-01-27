from setuptools import find_packages, setup

# print(find_packages())

setup(
    name="mldaikon",
    version="0.1",
    python_requires=">=3.8",
    packages=find_packages(),
    description="ML-DAIKON in development.",
    author="Yuxuan Jiang",
    author_email="jyuxuan@umich.edu",
    url="https://github.com/yourusername/your-project-name",
    install_requires=[
        "astor",
        "deepdiff",
        "polars>=1.0.0",
        "torch",
        "tqdm",
        "numba",
        "pandas",
        "pyyaml",
        "orjson",
    ],
)
