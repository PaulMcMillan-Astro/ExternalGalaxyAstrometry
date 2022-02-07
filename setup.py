
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="ExternalGalaxyAstrometry",
    version="0.1",
    author="Paul McMillan",
    author_email="paul@astro.lu.se",
    description="Tools to work with astrometric data for external galaxies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/PaulMcMillan-Astro/ExternalGalaxyAstrometry',
    python_requires='>=3',
    install_requires=["numpy","scipy","pandas"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"],
    packages=["ExternalGalaxyAstrometry"]
)