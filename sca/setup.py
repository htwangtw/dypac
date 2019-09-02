from setuptools import setup

setup(
   name='sdynca',
   version='0.2',
   description='stable dynamic cluster aggregation',
   author='Pierre Bellec',
   packages=['.'],  #same as name
   author_email='pierre.bellec@gmail.com',
   install_requires=['numpy', 'nilearn', 'sklearn'], #external packages as dependencies
)
