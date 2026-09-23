"""
Chess Engine in Python - Setup Configuration

A simple, type-safe chess engine for educational purposes.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess",
    version="0.1.0",
    author="Chess Engine Developer",
    description="A simple, type-safe chess engine in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/chess/issues",
        "Source Code": "https://github.com/yourusername/chess",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Games/Entertainment :: Video Games",
        "License :: OSI Approved :: MIT License",
        "Typing :: Typed",
    ],
    keywords="chess board game engine python typing",
    python_requires=">=3.8",
    package_dir={"": "chess"},
    packages=find_packages(where="chess"),
    install_requires=[],
)
