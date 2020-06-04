from setuptools import setup

setup(

    name='cellquantifier',
    version='0.0.1',
    description='CellQuantifier (a.k.a. cq) is a collection of algorithms \
                to get quantitative information of living cells',

    author='Clayton Seitz and Hua Lin',
    author_email='cwseitz@iu.edu',
    packages=['cellquantifier'],

    install_requires=['trackpy',
                     'pims',
                     'requests',
                     'scikit-image',
                     'seaborn',
                     'matplotlib-scalebar',
                     'matplotlib',
                     'pandas',
                     'scikit-learn',
                     'tensorflow-gpu==1.15.2',
                     'keras'],
)
