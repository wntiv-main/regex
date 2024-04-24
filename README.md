# Regex
Regular expression library using deterministic finite automaton (DFA)
to build and evaluate regular expressions efficiently.

## Environment
Requires Python 12

## Running examples
Module is required to be built and installed to run examples, after which these examples can simply be run
Either of the below methods will work to get the examples running

### Running module distribution
Requires `pip install build`
Task: `Deploy: Install module` to install module into python env
Task: `Uninstall module` to uninstall and cleanup
This distribution (output in dist/) *could* then be uploaded to a package index, e.g. PyPI
I have chosen *not* to do this (yet?) as this is a school project and not stable enough to
be used in production.

### Running without distribution
Faster, installs the module source directly
Task: `Develop: Install module` to install module into python env
Task: `Develop: Uninstall module` to uninstall and cleanup

## Running tests
`launch.json` provides two run configurations to output the tests
- Normal: Runs the tests on the regex module (requires that it is installed, see above)
- Distribution: Alternate config to automatically build and install the deployment distribution of the module,
				run the tests, and finally, automatically return to the develop install of the module. Requires
				`pip install build`
