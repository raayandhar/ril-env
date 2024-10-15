from setuptools import setup, find_packages

setup(
    name="ril-env",
    version="0.1.0",
    description="Environment and controller functionality and scripts",
    author="Raayan Dhar",
    author_email="raayan.dhar@gmail.com",
    packages=find_packages(include=["ril_env", "ril_env.*"]),
    install_requires=[],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
            "pip-tools",
        ]
    },
    python_requires=">=3.8",
    include_package_data=True,
)
