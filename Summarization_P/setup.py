# setup.py

from setuptools import setup, find_packages

setup(
    name="Summarization_P",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "langchain==0.3.13",
        "langchain-community==0.3.13",
        "langchain-huggingface==0.2.2",
        "langchain-ollama==0.2.2",
        "tiktoken==0.4.0",
        "beautifulsoup4==4.12.2",
        "requests==2.31.0",
        "gradio==5.9.1",
        "aiohttp==3.8.4",
        "python-dotenv==1.0.0",
        "elasticsearch==8.6.3",
        "sentence-transformers==2.2.2",
        "chardet==5.1.0",
        "pytest==7.4.0"
    ],
    python_requires=">=3.9",
)
