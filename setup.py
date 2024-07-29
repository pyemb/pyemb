from setuptools import setup

setup(name='pyemb',
      version='1.0.0-alpha',
          author='Annie Gray',
    author_email='annie.gray@bristol.ac.uk',
      description='EDA for complex data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
#     url='https://github.com/yourusername/your_project',
    # packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
      install_requires=['numpy', 'pandas']
      )
   