from setuptools import setup

setup(name='kanchay',
      version='0.2',
      description='Tools based on lightkurve, starspot and exoplanet codes to measure rotational periods',
      url='https://github.com/ramstojh/kanchay',
      author='Jhon Yana',
      author_email='ramstojh@alumni.usp.br',
      license='MIT',
      packages=['kanchay'],
      #install_requires=['matplotlib', 'tqdm', 'numpy', 'pandas', 'pymc3', 'theano', 'exoplanet'],
      zip_safe=False)
