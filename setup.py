from setuptools import setup, find_packages

setup(
    name="MAModule",
    version="1.0",
    keywords=("pip", "Multiagent"),
    description="A Module for multiagent tasks",
    long_description="A Module for multiagent tasks",
    license="MIT License",
    url="https://github.com/MrReochen/MultiAgentModule",
    author="Reo Chen",
    author_email="reo@pku.edu.cn",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        'numpy>=1.17.0',
        'torch',
        'PyBark>=1.0',
        'tensorboardx>=2.2',
        'rich>=10.12.0',
        'pyyaml>=5.4.1'
    ]
)