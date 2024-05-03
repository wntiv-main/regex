# Regex
Regular expression library using deterministic finite automaton (DFA)
to build and evaluate regular expressions efficiently.

## Environment
Requires Python 12
Requires `pip install numpy`

## Building manually
In project root directory:
```
python -m pip install build
python -m build
pip install ./dist/<name_of_build_wheel>.whl[<comma-seperated extras>]
```
Available extras: debuggraphviewer, copy-html
e.g.
- `pip install ./dist/<name_of_build_wheel>.whl`
- `pip install ./dist/<name_of_build_wheel>.whl[debuggraphviewer]`
- `pip install ./dist/<name_of_build_wheel>.whl[debuggraphviewer, copy-html]`

## VSCode Run Configurations
Module is required to be built and installed to run examples, after which these examples can simply be run
Build tasks are provided to do the above building for you. Note that these tasks do NOT install any extras,
if you need those either install the requirements manually, or follow the manual build instructions

### Running module in development
Faster, links the module source directly, also meaning that module doesnt have to be re-built and installed
whenever modifications are made to the module. Useful for development
Task: `Develop: Install module` to install module into python env
Task: `Develop: Uninstall module` to uninstall and cleanup

### Building module deployment distribution
Requires `pip install build`, slower
Task: `Deploy: Install module` to install module into python env
Task: `Uninstall module` to uninstall and cleanup
This distribution (output in dist/) *could* then be uploaded to a package index, e.g. PyPI
I have chosen *not* to do this (yet?) as this is a school project and not stable enough to
be used in production.

## Running tests
`launch.json` provides three run configurations to output the tests
- Normal: Runs the tests on the regex module (requires that it is installed, see above)
- Headless: Same as above, but does not show GUI for failed tests (does not require debuggraphviewer)
- Distribution: Alternate config to automatically build and install the deployment distribution of the module,
                run the tests, and finally, automatically return to the develop install of the module. Requires
                `pip install build`
There are also run configurations for:
- Playground: a workspace for testing
- Example executable: Runs the regex_viewer.py example using pyinstaller executable.
                      Requires `pip install pyinstaller`
