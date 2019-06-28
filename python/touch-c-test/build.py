import os
import torch
# from torch.utils.ffi import create_extension
from torch.utils.cpp_extension import CppExtension
from setuptools import setup
from setuptools import find_packages

sources = ['src/ext_lib.c']
headers = ['src/ext_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/ext_lib_cuda.c']
    headers += ['src/ext_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = CppExtension(
    '_ext.ext_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda
)

setup(
    name="test01",
    version="0.1",
    packages=find_packages(exclude=("configs", "tests",)),
    # install_requires=requirements,
    ext_modules=[ffi],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)