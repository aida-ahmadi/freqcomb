from setuptools import setup, find_packages

version = {}
with open('freqcomb/version.py') as v:
    exec(v.read(), version)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='freqcomb',
    version=version['__version__'],
    author='Aida Ahmadi',
    author_email='aahmadi@strw.leidenuniv.nl',
    description='Combing through frequency distribution of molecular line transitions',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['freqcomb']),
    url='https://github.com/aida-ahmadi/freqcomb',
    project_urls={
        "Bug Tracker": "https://github.com/aida-ahmadi/freqcomb/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    install_requires=['numpy>=1.15', 'pandas>1.0', 'matplotlib>=3.3.0',
                      'astropy>=3.1.2', 'astroquery>=0.4.2', 'seaborn',
                      'sklearn', 'scipy'],
    python_requires='>=3.6'
)
