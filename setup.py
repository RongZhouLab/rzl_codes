from setuptools import setup, find_packages

setup(
    name='RongZhouLabCodes',
    version='0.0.1',
    author="mromanelloj",
    packages=find_packages(),
    install_requires=[
        'numpy>1.20',
        'scipy>1.6',
        'matplotlib>3.4',
    ]
)
