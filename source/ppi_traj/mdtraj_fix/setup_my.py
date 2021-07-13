import sys
import os
import glob
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

#extensions = [Extension('my_rmsd', ['my_rmsd.pyx'] + os.listdir('src') + os.listdir('include'))]
extensions = [Extension('my_rmsd', ['my_rmsd.pyx'] + glob.glob('src/*.c') + glob.glob('src/*.cpp'))]

setup(
    name='my_rmsd',
    ext_modules=cythonize(
        Extension(
            'my_rmsd',
            sources=['my_rmsd.pyx'] + glob.glob('src/*.c') + glob.glob('src/*.cpp'),
            include_dirs=['include']
        )
    ),
    install_requires=["numpy"]
)

#setup(
    #ext_modules = cythonize(extensions, include_path=['include'])   
#)