import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='prosac',
    version='1.0',
    author='William Guimont-Martin',
    author_email='william.guimont-martin.1@ulaval.ca',
    description='PROSAC algorithm in python ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/willGuimont/PROSAC',
    project_urls={
        'Bug Tracker': 'https://github.com/willGuimont/PROSAC/issues'
    },
    license='MIT',
    packages=['prosac'],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
)
