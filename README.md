# 2 Room Thermostat Model

Dynamical model of 2 rooms heated by a thermostat.

The purpose of this model is to serve as an example for the python function instrumentation
approach implemented [here](https://gitlab.com/sbtg/instrumentation/branch-statement-analyzer)
along with the optimization approach implemented [here](https://gitlab.com/sbtg/pysoar-c).

## Usage

This project contains an entrypoint to run both a uniform random sampler and the SOAR-C sampler
for comparison. To execute this entrypoint, run the command `poetry run python3 -m instrumentation`
after setting up the poetry environment by executing `poetry install`.
