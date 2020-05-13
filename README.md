govuk-corona-analysis
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    │
    ├── Makefile                <- Makefile with commands like `make data` or `make train`
    │
    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── CONTRIBUTING.md         <- Guide to how potential contributors can help with your project
    │
    ├── .env                    <- Where to declare individual user environment variables
    │
    ├── .gitignore              <- Files and directories to be ignored by git
    │
    ├── test_environment.py     <- Python environment tester
    │
    ├── data
    │   ├── external             <- Data from third party sources.
    │   ├── interim              <- Intermediate data that has been transformed.
    │   ├── processed            <- The final, canonical data sets for modeling.
    │   └── raw                  <- The original, immutable data dump.
    │
    ├── docs                            <- A default Sphinx project; see sphinx-doc.org for details
    │   └── pull_request_template.md    <- Pull request template
    │
    ├── models                   <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                               the creator's initials, and a short `-` delimited description, e.g.
    │                               `1.0-jqp-initial-data-exploration`.
    │
    ├── references               <- AQA plan, Assumptions log, data dictionaries, and all other explanatory materials
    │   ├── aqa_plan.md          <- AQA plan for the project
    │   └── assumptions_log.md   <- where to log key assumptions to data / models / analyses
    │
    ├── reports                  <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures              <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                 <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                      <- Source code for use in this project.
        ├── __init__.py          <- Makes src a Python module
        │
        ├── make_data            <- Scripts to download or generate data
        │
        ├── make_features        <- Scripts to turn raw data into features for modeling
        │
        ├── make_models          <- Scripts to train models and then use trained models to make predictions
        │
        ├── make_visualisations  <- Scripts to create exploratory and results oriented visualizations
        │
        └── tools                <- Any helper scripts go here
       



--------

## Requirements

To run the code in this GitHub repository, please make sure your system meets the following requirements:

* Unix-like operating system (macOS, Linux, …);
* [`direnv`](https://direnv.net/) installed, including shell hooks;
* [`.envrc`](.envrc) allowed/trusted by `direnv` to use the environment variables - see
[below](#allowingtrusting-envrc);
* If missing, [create a `.secrets` file](#creating-a-secrets-file) to store untracked secrets;
* Python 3.5 or above; and
* [Python packages installed]() from the [`requirements-dev.txt`](requirements-dev.txt) file.

Note there may be some Python IDE-specific requirements around loading environment variables, which are not considered
here.

### Allowing/trusting `.envrc`

To allow/trust the [`.envrc`](.envrc) run the `allow` command using `direnv` at the top level of this repository.

```shell script
direnv allow
```

### Creating a `.secrets` file

Secrets used by this repository can be stored in a `.secrets` file. **This is not tracked by Git**, and so secrets will
not be committed onto your remote.

In your shell terminal, at the top level of the repository, create a `.secrets` file.

```shell script
touch .secrets
```

Open this new `.secrets` file using a text editor, and add any secrets as environmental variables. For example, to add
a JSON credentials file for Google BigQuery, add the following code to `.secrets`.

```shell script
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### Installing Python packages

This repository uses Python packages for both analytical and software development purposes. For the latter, it uses
pre-commit hooks to ensure best practice when committing code to this repository; further details are given
[below](#pre-commit-hooks).

This code uses Python packages that depend on [International Components for Unicode](http://site.icu-project.org/home)
C/C++ and Java libraries for Unicode and globalization, such as [`PyICU`](https://pypi.org/project/PyICU/) and
[`polyglot`](https://pypi.org/project/polyglot/); these ICU libraries need to be installed, and added to `PATH` before
the Python packages are installed, otherwise `pip` will fail. To do this on MacOS using [Homebrew](https://brew.sh/):

```shell script
brew install icu4c
export PATH="/usr/local/opt/icu4c/bin:$PATH"
```

To install the required Python packages via `pip`, first set up a Python virtual environment; this ensures you do not
install the packages globally. Once you have activated your virtual environment, install the Python packages from
[`requirements-dev.txt`](requirements-dev.txt).

```shell script
pip install -r requirements-dev.txt
```

Finalise the installation by setting up the pre-commit hooks whenever code is pushed.

```shell script
pre-commit install -t pre-push
```

## Building and viewing documentation

The documentation for this project is build using Sphinx, and resides in the `docs` folder. To generate a copy of the
documentation locally, in your terminal from the top-level of this repository run the following commands:

```shell script
cd docs
make clean
make html
```

Use the same commands to rebuild your documentation if there are any updates. Once built, the documentation can be
view in your local browser at [`docs/_build/html/index.html`](docs/_build/html/index.html).

##  Pre-commit hooks



This repo uses the Python package `pre-commit` (https://pre-commit.com) to manage pre-commit hooks. Pre-commit hooks are

actions which are run automatically, typically on each commit, to perform some common set of tasks. For example, a pre-commit

hook might be used to run any code linting automatically, providing any warnings before code is committed, ensuring that

all of our code adheres to a certain quality standard.



For this repo, we are using `pre-commit` for a number of purposes:

- Checking for AWS or private access keys being committed accidentally

- Checking for any large files (over 5MB) being committed

- Cleaning Jupyter notebooks, which means removing all outputs and execution counts

- Running linting on the `src` directory (catching problems before they get to Concourse, which runs the same check)



We have configured `pre-commit` to run automatically _when pushing_ rather than on _every commit_, which should mean we

receive the benefits of `pre-commit` without it getting in the way of regular development.



In order for `pre-commit` to run, action is needed to configure it on your system; see the
[Installing Python packages](#installing-python-packages) section for further details.



###  Note on Jupyter notebook cleaning



It may be necessary or useful to keep certain output cells of a Jupyter notebook, for example charts or graphs visualising

some set of data. To do this, add the following comment at the top of the input block:



`# [keep_output]`



This will tell `pre-commit` not to strip the resulting output of this cell, allowing it to be committed.

## Running tests

Tests for this GitHub repository are written in [`pytest`](https://docs.pytest.org/en/latest/). To execute the tests,
run the following command:

```shell script
pytest
```

### Test coverage

To view the test coverage within the `src` folder as a HTML report, run the following commands:

```shell script
coverage run -m pytest
coverage html
```

Once complete, you can view the coverage report in your local browser at [`htmlcov/index.html`](htmlcov/index.html).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
