Benchmarking
===============================================================

WARNING: benchmarking is not working

Here is info about benchmarking various parts of Dinora engine.
Benchmarking needed to improve chess engine speed, which leads to
producing better moves in less time.

Running benchmarks
------------------

Dinora uses pytest and pytest-benchmark for this task.
All benchmarks are stored under tests/bench

.. code-block:: text

    $ pytest tests/bench

Improve engine speed
--------------------

Before you start working on improvements. Capture current speed

.. code-block:: text

    $ pytest tests/bench --benchmark-save=<your-name>

It will save benchmark as <some_number>_<your-name>.json
Write some code and run benchmark again

.. code-block:: text

    $ pytest tests/bench --benchmark-save=<my-improvements>

Also saved as <new_number>_<my_improvements>.json
After this you can compare benchmark results

.. code-block:: text

    $ pytest-benchmark compare <some_number> <new_number>
