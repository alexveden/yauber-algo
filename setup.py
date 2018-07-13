from setuptools import setup

setup(name='yauber-algo',
      version='0.1a',
      description='Financial time-series toolbox',
      url='https://github.com/alexveden/yauber-algo',
      author='Alex Veden',
      author_email='i@alexveden.com',
      license='MIT',
      packages=['yauber_algo'],
      install_requires=[
            'pandas',
            'numpy',
            'numba',
      ],
      zip_safe=False)