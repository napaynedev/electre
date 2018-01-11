from setuptools import setup
setup(
    name = 'electre',
    version = '0.1.0',
    packages = ['electre'],
    entry_points = {
        'console_scripts': [
            'electre = electre.__main__:main'
        ]
    })
