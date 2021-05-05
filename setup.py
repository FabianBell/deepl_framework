#!/usr/bin/python
import setuptools

with open('README.md', 'r') as fin:
    long_description = fin.read()

with open('requirements.txt', 'r') as fin:
    requirements = fin.read().split('\n')

setuptools.setup(name='deep-nlp',
    version='0.0.1',
    description='Deep nlp library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Fabian Bell',
    author_email='fabianx.bell@gmail.com',
    url='https://github.com/FabianBell/deepl_framework',
    license='MIT',
    packages=setuptools.find_packages(),
    scripts=['scripts/extract_model'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=requirements,
    python_required='>=3.8'
    )
