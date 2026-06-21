from setuptools import setup, find_packages

setup(
    name="devmind",
    version="1.0.0",
    description="AI-powered debugging copilot — execution time machine + infrastructure analysis",
    author="Nitaiz123",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn>=0.29.0",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-asyncio", "httpx"],
    },
    entry_points={
        "console_scripts": [
            "devmind=devmind.cli.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Debuggers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
