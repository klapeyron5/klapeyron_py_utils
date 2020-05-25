import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='klapeyron_py_utils',
    version='0.8.2',
    author="Nikita Klapeyron",
    author_email="nikitaklapeyron@gmail.com",
    description="Just my python reusable code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klapeyron5/klapeyron_py_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
)
