from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="dbgsom",
    version="1.1.0",
    description="A Python implementation of the Directed Batch Growing Self-Organizing Map.",
    long_description=long_description,
    author="Sandro Martens",
    author_email="sandro.martens@web.de",
    license="MIT",
    packages=["dbgsom"],
    install_requires=[
        "numpy",
        "networkx",
        "pandas",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "numpydoc",
    ],
    url="https://github.com/SandroMartens/DBGSOM"
)
