from setuptools import setup, find_packages

setup(name='yauber_algo',
      version='0.3a',
      description='Financial time-series toolbox',
      url='https://github.com/alexveden/yauber-algo',
      author='Alex Veden',
      author_email='i@alexveden.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'numpy',
            'numba',
      ],
      zip_safe=False)