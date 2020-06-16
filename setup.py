#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ['requirements.txt']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Aviana Global Technologies",
    author_email='info@avianaglobal.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Nemesis anomaly detection system",
    entry_points={
        'console_scripts': [
            'nemesis2=nemesis2.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    keywords='nemesis-ml',
    name='nemesis-ml',
    packages=find_packages(include=['nemesis-ml', 'nemesis-ml.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AvianaGlobal/nemesis2',
    version='0.1.0',
    zip_safe=False,
)
