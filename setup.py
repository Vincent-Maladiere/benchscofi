from setuptools import setup, find_packages

NAME = "benchscofi"
VERSION = "9999"

setup(name=NAME,
    version=VERSION,
    author="Clémence Réda",
    author_email="recess-project@proton.me",
    url="https://github.com/RECeSS-EU-Project/benchscofi",
    license_files = ('LICENSE'),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords='',
    description="Package which contains implementations of published collaborative filtering-based algorithms for drug repurposing.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={'':"src"},
    python_requires='>=3.8.5',
    install_requires=[
        #"stanscofi>=2.0.0",
        #"tensorflow==2.4.3",
        #"pulearn>=0.0.7",
        "torch",
        #"fastai>=2.7.12",
        "torch_geometric",
        #"pyFFM",
        #"protobuf==3.9.*",
        #"pydantic==1.9.*",
        "pytorch-lightning",
        "torchmetrics",
        #"numpy",
        #"scikit-learn",
        "dgl",
        #"libmf>=0.9.2",
    ],
    entry_points={},
)
