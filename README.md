# ARC-phash

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

An investigatory repo exploring the relationship between perceptual hashing and in-painting

## Installation

```bash
python -m pip install arc_phash
```

From source:
```bash
git clone https://github.com/alan-turing-institute/ARC-phash
cd ARC-phash
python -m pip install .
```

## Usage

Ensure you have docker installed and running, then build the image with e.g.

```shell
docker build -t my_pipeline .
```

and run it with

```shell
docker run --rm --volume .:/app my_pipeline
```

where we have used `--volume` to create a bind mount which will save off a database named `embeddings.db` locally for querying and EDA. For example, run

```shell
sqlite3 embeddings.db
```

Then, to view a sample of rows:

```sql
SELECT * FROM embeddings LIMIT 10;
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-phash/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-phash/actions
[pypi-link]:                https://pypi.org/project/ARC-phash/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/ARC-phash
[pypi-version]:             https://img.shields.io/pypi/v/ARC-phash
<!-- prettier-ignore-end -->
