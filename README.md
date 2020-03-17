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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



##  Installing pre-commit hooks

  

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

  

In order for `pre-commit` to run, action is needed to configure it on your system.

  

- Run `pip install -r requirements-dev.txt` to install `pre-commit` in your Python environment

- Run `pre-commit install -t pre-push` to set-up `pre-commit` to run when code is _pushed_

  

###  Note on Jupyter notebook cleaning

  

It may be necessary or useful to keep certain output cells of a Jupyter notebook, for example charts or graphs visualising

some set of data. To do this, add the following comment at the top of the input block:

  

`# [keep_output]`

  

This will tell `pre-commit` not to strip the resulting output of this cell, allowing it to be committed.
