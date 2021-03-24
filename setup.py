from setuptools import setup, find_packages

with open('README.md', 'r') as f:
	long_description = f.read()

with open('requirements.txt') as f:
  requirements = f.read().splitlines()

setup(
	name='sklearn-benchmarks',
	packages=find_packages(),
	entry_points={'console_scripts': ['sklbench = sklearn-benchmarks.cli.__main__:main', 'sklearn-benchmarks = sklearn-benchmarks.cli.__main__:main']},
	version='0.1.0',
	description='Benchmarking tool to compare sklearn\'s performances with other librairies',
	long_description=long_description,
	py_modules=['sklearn-benchmarks'],
	install_requires=requirements
)
