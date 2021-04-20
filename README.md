# PeekingDuck

<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/aimakerspace/PeekingDuck?style=for-the-badge" style="margin: 0px 8px  20px 0px">
<img alt="GitHub all releases" src="https://img.shields.io/github/downloads/aimakerspace/PeekingDuck/total?style=for-the-badge" style="margin: 0px 8px  20px 0px">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/aimakerspace/PeekingDuck?style=for-the-badge" style="margin: 0px 8px  20px 0px">

PeekingDuck is a python framework dealing with model inference.

This toolkit provides state of the art computer vision models to make real time inference easy: *social distancing, people counting, license plate recognition etc.*. Customisability of the node modules ensures flexibility for unique usecases. Attached API server enables real time access to data analysis of the model pipe.

## Features

### Models

- Yolov4
- EfficientDet
- Blazepose

### Use Cases

- Social Distancing
- People Counting
- License Plate Recognition
- Vehicle Counting
- Privacy Filter

## How to Use for Developers (Temporary)

- git clone this repo
- Choose the required nodes in [run_config.yml](run_config.yml)
- To run:
    ```
    cd ..
    python peekingduck
    ```
- To create a new node, check out [CONTRIBUTING.md](CONTRIBUTING.md)




## Installation (WIP)

Use python package manager (pip) to install PeekingDuck

`pip install pkdk`

## Usage (WIP)

### Start New Projects with `init`
For new projects, we suggest to use the PeekingDuck cookiecutter starter:

```bash
> mkdir <new_project>
> cd <new_project>
> peekingduck init
```

### `get-configs`
Unless specified, all nodes in `peekingduck` will use the default configs for every node. To view and change these configs, you can use:

``` bash
> peekingduck get-configs
```

For specific information on how to use peekingduck-cli, you can use `peekingduck --help`.

## Contributing (WIP)

We welcome contributions to the repository through pull requests. When making contributions, first create an issue to describe the problem and the intended changes.

Please note that we have a code of conduct for contributions to the repository.

## License

Licensed under [Apache License, Version 2.0](LICENSE)

