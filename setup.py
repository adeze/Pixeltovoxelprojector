from setuptools import setup

setup(
    name='pixeltovoxelprojector',
    version='0.1.0',
    description='Pure Python pipeline for pixel-to-voxel projection and visualization',
    author='adeze',
    packages=[],  # Add your package/module names here if needed
    install_requires=[
        'numpy',
        'torch',
        'pillow',
        'pyvista',
        'astropy',
        'matplotlib',
    ],
)
