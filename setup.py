 from setuptools import setup

 setup(
   name='TightFrames',
   version='0.1.0',
   author='Maciej Żurawski',
   author_email='mac@example.com',
   packages=['numpy', 'scipy'],
   scripts=['bin/script1','bin/script2'],
   # url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='An awesome package that does something',
   long_description=open('README.txt').read(),
   install_requires=[
       # "Django >= 1.1.1",
       "pytest",
   ],
)