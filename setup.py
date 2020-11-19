from setuptools import setup, find_packages
import sys

setup(
    name="DSAE-Impute",
    version="1.1.4",
    author="Huan Deng",
    author_email="lehaftcode@gmail.com",
    description="scRNA-seq imputation",
    long_description=" Learning Discriminative Stacked Autoencoders for Imputing Single-cell RNA-seq Data. ",
    keywords=['scRNA-seq', 'imputation', 'stacked autoencoder',
              'discriminative', 'single-cell'],
    license="MIT License",
    url="https://github.com/lebhaftcode/DSAE-Impute.git",
    packages=['DSAE_Impute','DSAE_Impute.DSAE','DSAE_Impute.data'],
    entry_points={   
        'console_scripts': [
            'DSAE_Impute=DSAE_Impute.__main__:main'
        ]},
    data_files= [('DSAE_Impute/data',['./DSAE_Impute/data/test.csv'])]
)
