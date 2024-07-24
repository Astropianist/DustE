import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DustE",                     # This is the name of the package
    version="0.0.2",                        # The initial release version
    author="Gautam Nagaraj",                     # Full name of the author
    author_email="gxn75@psu.edu",
    description="Calculate dust attenuation curves as function of physical properties",
    project_urls={"Source repo": "https://github.com/Astropianist/DustE"},
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where='src'),    # List of all python modules to be installed
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],                                      # Information to filter the project on PyPi website
    license='MIT',
    package_data={"": ["README.md", "LICENSE", "DustAttnCurve_bv_1_eff_0_01.png"], "duste": ["Marg/*.dat", "TraceFiles/*.dat", "TraceFiles/*.nc"]},
    include_package_data=True,
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=["numpy","scipy","matplotlib","seaborn","astropy","arviz","astro-sedpy", "dynesty"]  # Install other dependencies if any
)
