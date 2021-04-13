import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="equityboard", # Replace with your own username
    version="0.0.1",
    author="Anselme Borgeaud",
    author_email="afeborgeaud@gmail.com",
    description="Dashboard simple portfolio analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/afeborgeaud/equityboard",
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "equityboard"},
    packages=setuptools.find_packages(where="equityboard"),
    python_requires=">=3.9",
    requires=[
        "requests",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "dash",
        "plotly",
        "pywavelets",
        "tqdm",
        "pyarrow",
    ]
)
