# setup.py
from setuptools import setup, find_packages

setup(
    name="special_attentions",
    version="0.2",
    packages=find_packages(include=['*', 'utils.*']),  # 显式包含所有层级
    package_dir={"": "."},  # 声明当前目录为包根
    install_requires=[]
)