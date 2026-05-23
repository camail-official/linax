<div align="center">

# Discretax - State Space Models in JAX

<img src="https://raw.githubusercontent.com/camail-official/discretax/main/assets/logo.png" alt="Discretax logo" width="200"/>


[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/camail-official/discretax/actions/workflows/tests.yml/badge.svg)](https://github.com/camail-official/discretax/actions/workflows/test.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/camail-official/discretax/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)[![JAX](https://img.shields.io/badge/JAX-0.7%2B-0A7AAA?logo=jax&logoColor=white)](https://github.com/google/jax)
[![PyPI version](https://img.shields.io/pypi/v/discretax)](https://pypi.org/project/discretax/)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/VazrGCxeT7)

</div>

[discretax](https://github.com/camail-official/discretax) is a collection of state space models implemented in JAX. It is

- easy to use
- fast
- modular

## Table of contents

- [Discretax - State Space Models in JAX](#discretax---state-space-models-in-jax)
  - [Table of contents](#table-of-contents)
  - [News](#news)
  - [Just get me Going](#just-get-me-going)
  - [Join the Community](#join-the-community)
  - [Installation](#installation)
    - [Full Library Installation](#full-library-installation)
  - [Supported Models](#supported-models)
  - [Contributing](#contributing)
  - [Core Contributors](#core-contributors)
  - [Citation](#citation)

## News

- [2026-03]: After a big refactor, we are renaming the project from linax to discretax.
- [2025-10]: We are happy to launch the first beta version of linax. 🎉

## Just get me Going

If you don't care about the details, we provide [example notebooks](examples/) that are ready to use.

## Join the Community

To join our growing community of JAX and state space model enthusiasts, join our [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/VazrGCxeT7) server. Feel free to write us a message (either there or to our personal email, see the bottom of this page) if you have any questions, comments, or just want to say hi!

🤫 Psssst! Rumor has it we are also developing an end-to-end JAX training pipeline. Stay tuned for JAX Lightning. So join the discord server to be the first to hear about our newest project(s)!

## Installation

[discretax](https://github.com/camail-official/discretax) is available as a PyPI package. To install it via uv, just run

```bash
uv add discretax
```

or

```bash
uv add discretax[cu12]
```

If pip is your package manager of choice, run

```bash
pip install discretax
```

or

```bash
pip install discretax[cu12]
```

### Full Library Installation

If you want to install the full library, especially if you want to **contribute** to the project, clone the [discretax](https://github.com/camail-official/discretax) repository and cd into it

```bash
git clone https://github.com/camail-official/discretax.git
cd discretax
```

If you want to install dependencies for CPU, run

```bash
uv sync
```

for GPU run

```bash
uv sync --extra cu12
```

To include development tooling (pre-commit, Ruff), install:

```bash
uv sync --extra dev
```

After installing the development dependencies (activate your environment if needed), enable the git hooks:

```bash
uv run pre-commit install
```

## Supported Models

| Year | Model  | Paper                                                                                                                                                                                         | Code                                                                      | Our implementation                                                                                       |
| ---- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 2024 | DeltaNet | [Parallelizing Linear Transformers with the Delta Rule](https://openreview.net/forum?id=y8Rm4VNRPH)                                                                                                    | [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/deltanet.py)    |
| 2024 | LinOSS | [Oscillatory State Space Models](https://openreview.net/pdf?id=GRMfXcAAFh)                                                                                                                    | [tk-rusch/linoss](https://github.com/tk-rusch/linoss)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/linoss.py)       |
| 2023 | LRU    | [Resurrecting Recurrent Neural Networks for Long Sequences](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)                                                                     | [LRU paper](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf) | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/lru.py)          |
| 2022 | S5     | [Simplified State Space Layers for Sequence Modeling](https://openreview.net/pdf?id=Ai8Hw3AXqks)                                                                                              | [lindermanlab/S5](https://github.com/lindermanlab/S5)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/models/s5.py)           |
| 2022 | S4D    | [On the Parameterization and Initialization of Diagonal State Space Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/e9a32fade47b906de908431991440f7c-Paper-Conference.pdf) | [state-spaces/s4](https://github.com/state-spaces/s4)                     | [discretax](https://github.com/camail-official/discretax/blob/main/src/discretax/sequence_mixers/s4d.py) |

## Contributing

If you want to contribute to the project, please check out [contributing](docs/contributing.md)

## Core Contributors

This repository has been created and is maintained by:

- [Philipp Nazari](https://phnazari.github.io)
- [Francesco Maria Ruscio](https://github.com/francescoshox)
- [Benedict Armstrong](https://github.com/benedict-armstrong)

The LinOSS updates were contributed by [Kasra Mazaheri](https://github.com/KasraMazaheri)
and [Jared Boyer](https://github.com/jaredbmit).

This work has been carried out within the [Computational Applied Mathematics & AI Lab](https://camail.org),
led by [T. Konstantin Rusch](https://github.com/tk-rusch).

## Citation

If you find this repository useful, please consider citing it.

```bib
@software{discretax2025,
  title  = {Discretax: A Lightweight Collection of State Space Models in JAX},
  author = {Nazari, Philipp* and Ruscio, Francesco Maria* and Armstrong, Benedict* and Rusch, T. Konstantin},
  url    = {https://github.com/camail-official/discretax},
  year   = {2025}
}
```
