from setuptools import setup, find_packages

setup(
    name="rag-project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        "langchain",
        "llama-index",
        # ... other dependencies
    ],
) 