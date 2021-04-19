from setuptools import setup

# Version
version = None
with open("compositional/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in compositional/__init__.py"

setup(
name='compositional',
    version=version,
    description='Compositional data analysis in Python',
    url='https://github.com/jolespin/compositional',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["compositional"],
    install_requires=[
        "pandas",
        "numpy",
      ],
)
