"""Fuzz the dimensionality estimators with Atheris.

Any 2D float matrix the fuzzer builds must either be rejected with ValueError
or give the four metrics without raising anything else.

Run locally (Linux, Python 3.12):
    pip install --require-hashes -r .github/requirements/fuzz.txt
    pip install --no-deps -e .
    python fuzz/fuzz_estimators.py -max_total_time=60
"""

import sys

import atheris
import torch

with atheris.instrument_imports(include=["ndt"]):
    from ndt.core.estimators import compute_all_metrics
    from ndt.core.estimators import cumulative_energy_90
    from ndt.core.estimators import nuclear_norm_ratio
    from ndt.core.estimators import participation_ratio
    from ndt.core.estimators import stable_rank

ESTIMATORS = (
    stable_rank,
    participation_ratio,
    cumulative_energy_90,
    nuclear_norm_ratio,
    compute_all_metrics,
)


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    rows = fdp.ConsumeIntInRange(0, 16)
    cols = fdp.ConsumeIntInRange(0, 16)
    values = [fdp.ConsumeRegularFloat() for _ in range(rows * cols)]
    matrix = torch.tensor(values, dtype=torch.float32).reshape(rows, cols)
    estimator = ESTIMATORS[fdp.ConsumeIntInRange(0, len(ESTIMATORS) - 1)]
    try:
        estimator(matrix)
    except ValueError:
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
