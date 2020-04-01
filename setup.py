from setuptools import setup

setup(

    name='cellquantifier',
    version='0.0.1',
    description='CellQuantifier (a.k.a. cq) is a collection of algorithms \
                to get quantitative information of living cells',

    author='Clayton Seitz',
    author_email='cwseitz@iu.edu',
    packages=['cellquantifier'],

    install_requires=['trackpy==0.4.2',
                     'pims==0.4.1',
                     'scikit-image==0.16.2',
                     'seaborn==0.9.0',
                     'matplotlib-scalebar==0.6.1',
                     'matplotlib',
                     'pandas==0.25.3',
                     'scikit-learn'],
)
