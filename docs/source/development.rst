Development
===========

pre-commit
----------
This project uses pre-commit https://pre-commit.com/
After cloning repository and installing poetry dependecies,
install git hooks by

.. code-block:: console

    pre-commit install


mypy
----
To check python types with mypy http://mypy-lang.org/ you can run

.. code-block:: console

    mypy dinora

You also can run mypy in strict mode

.. code-block:: console

    mypy dinora --strict

So far not the whole project supports strict mypy,
you can submit Pull Request to help

pytest
------
To run tests

.. code-block:: console

    pytest

And generate html report

.. code-block:: console

    pytest --cov dinora --cov-report html

Now you can see report in htmlcov/index.html

Tests contribution are always appreciated
