# Machine Learning

To run the python scripts present in this repository, the following must be installed in your system.

- Python (>= 3.12) or higher
- `pip` (package manager for python)

## Virtual Environment

To run the project locally, setup a virtual environment on the home directory of the project. To create a `venv` in python, run the following command.

```bash
python -m venv ../mlforge
```

This defines a virtual environment in the home directory with the global python version being setup as the value for the venv. Once its been created, activate the venv by running the following command.

```bash
source ./bin/activate
```

To check if the venv was activated successfully, run the below command.

```bash
which python
```
This should point to the python executable inside the virtual environment. Once this activation is done, install the following dependencies:

```bash
pip install -r requirements.txt
```

The file `requirements.txt` contains all the direct dependencies and their version. To deactivate the virtual environment, run the below command.

```bash
deactivate
```