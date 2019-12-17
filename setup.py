import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='klapeyron_py_utils',
    version='0.1.7',
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
)
