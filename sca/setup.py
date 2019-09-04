from setuptools import setup

setup(
   name='dypac',
   version='0.3',
   description='dynamic parcel aggregation with clustering',
   author='Pierre Bellec',
   packages=['.'],  #same as name
   author_email='pierre.bellec@gmail.com',
   install_requires=['numpy', 'nilearn', 'sklearn'], #external packages as dependencies
)
