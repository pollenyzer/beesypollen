import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beesypollen",
    version="0.0.1",
    author="Parzival",
    author_email="parzival.borlinghaus@kit.edu",
    install_requires=[],
    description="Tools for pollen detection and pollen color extration on for pollen trap content.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["beesypollen"]),
    python_requires=">=3.6",
)
