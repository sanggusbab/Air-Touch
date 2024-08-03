from setuptools import setup, find_packages

setup(
    name='eye_tracking',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'mediapipe',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'eye_tracking=eye_tracking.main:main',
        ],
    },
)
