from setuptools import setup

setup(name='pyemb',
      version='1.0.0a3',
      author='Annie Gray',
      author_email='annie.gray@bristol.ac.uk',
      description='EDA for complex data',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/pyemb/pyemb',
      packages=['pyemb'],
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6',
      install_requires=['numpy', 'pandas']
      )
   