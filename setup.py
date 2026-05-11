import os
from setuptools import setup
script_directory = os.path.abspath(os.path.dirname(__file__))


# Version
version = None
with open("compositional/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in compositional/__init__.py"

requirements = list()
with open(os.path.join(script_directory, 'requirements.txt')) as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            if not line.startswith("#"):
                requirements.append(line)

setup(
name='compositional',
    version=version,
    description='Compositional data analysis in Python',
    url='https://github.com/jolespin/compositional',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["compositional"],
    install_requires=requirements,
)
