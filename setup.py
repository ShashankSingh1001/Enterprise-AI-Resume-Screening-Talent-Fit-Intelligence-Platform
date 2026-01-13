from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = this_directory / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            # Filter out comments and empty lines
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

setup(
    # Package Metadata
    name="resume-ai",
    version="1.0.0",
    author="Shashank Singh",
    author_email="shashanksinghofficial101@gmail.com",
    description="Enterprise AI Resume Screening & Talent Fit Intelligence Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShashankSingh1001/Enterprise-AI-Resume-Screening-Talent-Fit-Intelligence-Platform",
    
    # Package Configuration
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    include_package_data=True,
    
    # Python Version
    python_requires=">=3.11",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional Dependencies
    extras_require={
        'dev': [
            'pytest>=8.0.0',
            'pytest-cov>=4.1.0',
            'black>=24.1.1',
            'flake8>=7.0.0',
            'mypy>=1.8.0',
            'ipython>=8.12.0',
            'jupyter>=1.0.0',
        ],
        'deployment': [
            'gunicorn>=21.2.0',
            'docker>=7.0.0',
        ],
        'cloud': [
            'boto3>=1.34.0',  # AWS
            'google-cloud-storage>=2.14.0',  # GCP
            'azure-storage-blob>=12.19.0',  # Azure
        ],
        'monitoring': [
            'sentry-sdk>=1.40.0',
            'prometheus-client>=0.19.0',
        ],
    },
    
    # Package Data
    package_data={
        'config': ['*.yaml', '*.yml'],
        'src': ['logging/*.py', 'exceptions/*.py'],
    },
    
    # Entry Points (CLI commands)
    entry_points={
        'console_scripts': [
            'resume-ai-train=src.training.train_model:main',
            'resume-ai-predict=src.inference.predictor:main',
            'resume-ai-api=api.main:start_server',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Office/Business :: Human Resources',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Natural Language :: English',
    ],
    
    # Keywords
    keywords=[
        'resume-screening',
        'recruitment',
        'ai',
        'nlp',
        'machine-learning',
        'explainable-ai',
        'bias-detection',
        'fairness',
        'mlops',
        'fastapi',
        'streamlit',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/ShashankSingh1001/Enterprise-AI-Resume-Screening-Talent-Fit-Intelligence-Platform/wiki',
        'Source': 'https://github.com/ShashankSingh1001/Enterprise-AI-Resume-Screening-Talent-Fit-Intelligence-Platform',
        'Tracker': 'https://github.com/ShashankSingh1001/Enterprise-AI-Resume-Screening-Talent-Fit-Intelligence-Platform/issues',
    },
    
    # License
    license='MIT',
    
    # Zip Safe
    zip_safe=False,
)