from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = []
    for line in f:
        req = line.split('#')[0].strip()
        install_requires.append(req)
setup(
    name='kinatt',
    version='0.0.1',
    description='kinematic attractor for robot control',
    author='Hirokazu Ishida',
    author_email='h-ishida@jsk.imi.i.u-tokyo.ac.jp',
    license='MIT',
    install_requires=install_requires,
    packages=find_packages(exclude=('tests', 'docs'))
)
