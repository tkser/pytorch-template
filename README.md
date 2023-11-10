# pytorch_template

## Requirements

- Python 3.10
- Poetry

## Usage

### Prepare

```shell
sed -i 's/pytorch_template/<your-project-name>/g' pyproject.toml
mv -rf pytorch_template <your-project-name>

poetry install
```

And then, edit config files in `configs/` directory.

### Train

```shell
poetry run python train.py
```

### Test

```shell
poetry run python test.py --ckpt_path <path-to-checkpoint>
```
