import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torchgadgets',
    version='0.0.0',
    author='Luis Denninger',
    author_email='Luis0512@web.de',
    description='Initial State',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'torch', 'torchvision', 'matplotlib', 'ipdb', 'tqdm', 'tensorboard', 'fvcore'],
)