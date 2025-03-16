# ARC Perceptual Hash

An investigatory repo exploring the relationship between perceptual hashing and in-painting

## Installation

### Dev

Ensure you have uv installed then run

```bash
uv sync --all-extras
```

Install the pre-commit hooks with

```shell
uv run pre-commit install
```

and run with

```shell
uv run pre-commit run --all-files
```

## Usage

### Dataset generation

See [separate README](src/arc_phash/data_generation/README.md).

Example usage (will download model if not run before):

```shell
uv run src/arc_phash/data_generation/perform_inpainting.py people runwayml/stable-diffusion-inpainting
```

### Pipeline prototype

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
