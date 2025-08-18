from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hand-gesture-recognition",
    version="1.0.0",
    author="John Fields",
    author_email="jackdalefields@gmail.com",
    description="Hand gesture recognition for input control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johndfields/wow-hand-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
        "numpy>=1.19.0",
        "pynput>=1.7.0",
        "jsonschema>=3.2.0",
        "watchdog>=2.1.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "windows": ["vgamepad>=0.0.8"],
    },
    entry_points={
        "console_scripts": [
            "hand-gesture-recognition=src.main:main",
        ],
    },
)

