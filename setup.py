from setuptools import find_packages, setup

setup(
    name='nucleotran',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    version='0.1.0',
    python_requires=">=3.9",
    description='Efficient transformers for long nucleotide sequences',
    author='HPI Digital Health - Machine Learning',
    license='MIT',
)
