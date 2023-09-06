from distutils.core import setup


with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="EventDetector",
    version="0.1dev",
    packages=["eventdetector"],
    install_requires=requirements,
)
