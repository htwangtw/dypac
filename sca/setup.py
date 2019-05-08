from setuptools import setup

setup(
   name='sca',
   version='0.1',
   description='stable cluster aggregation',
   author='Pierre Bellec',
   author_email='pierre.bellec@gmail.com',
   packages=['.'],  #same as name
   install_requires=['numpy', 'sklearn'], #external packages as dependencies
)
