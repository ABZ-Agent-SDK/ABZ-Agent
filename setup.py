from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="abz-agents",
    version="0.5.4",
    description="The fastest way to build AI agents using Google Gemini & Groq",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["abzagent*"]),
    include_package_data=True,
    install_requires=[
        "google-genai",
        "pydantic>=2.6.0",
        "rich>=13.7.0",
        "python-dotenv>=1.0.1",
        "tzdata>=2024.1; platform_system == 'Windows'",
        "typing-extensions>=4.12.0; python_version < '3.11'",
        "groq"
    ],
    python_requires=">=3.10",
    license="MIT",
    author="Abu Bakar",
    url="https://github.com/ABZAgent/abz-agents",
    project_urls={
        "Homepage": "https://abzagent.online/",
        "Issues": "https://github.com/ABZAgent/abz-agents/issues",
        "Documentation": "https://abzagent.online/docs/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=["agents", "gemini", "google generative ai", "sdk", "llm", "tool calling" , "groq"],
     entry_points={
        "console_scripts": [
            "abz-agents=abzagent.cli:main",
        ],
     },
)
