from setuptools import setup, find_packages


setup(
    name='planetary-seismic-ai',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'torch',
        'obspy',
        'emd',
        'scikit-learn',
        'tqdm'
    ]
)