from setuptools import setup


def read_requirements():
    with open("requirements.txt", "r") as req:
        content = req.read()
        requirements = content.split("\n")

    return requirements


setup(
    name="pyemb",
    version="1.0.0a7",
    author="Annie Gray",
    author_email="annie.gray@bristol.ac.uk",
    description="EDA for complex data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pyemb/pyemb",
    packages=["pyemb"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=read_requirements(),
)
