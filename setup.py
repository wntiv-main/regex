"""Setup module"""

from setuptools import setup

setup(
    name='regex',
    version='0.0.1',
    description="Regular expression library using deterministic finite "
    "automaton (DFA) to build and evaluate regular expressions "
    "efficiently",
    install_requires=['numpy'],
    extras_require={
        "DebugGraphViewer": [
            "matplotlib",
            "networkx",
            "scipy",
            "PyQt5"],
        "_copy_html": ["pywin32"]
    },
    license='None',
    packages=['regex'],
    author='Callum Hynes',
    author_email='hynescj20@cashmere.school.nz',
    keywords=['regular expression', 'DFA'],
    url='https://github.com/wntiv-main/regex')
