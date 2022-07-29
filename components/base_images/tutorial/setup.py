from setuptools import find_packages, setup

setup(
    name="forecasting_base_image",
    version="0.0.1",
    description="Forecasting Base image used to build components on top of it",
    packages=find_packages(),
    include_package_data=True,
)
