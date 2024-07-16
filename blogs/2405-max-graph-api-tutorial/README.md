# MAX Graph API Tutorial

Published blog post: [MAX Graph API Tutorials](https://www.modular.com/blog/max-graph-api-tutorial)

Versions used in this blog

```txt
max 24.3.0 (9882e19d)
Modular version 24.3.0-9882e19d-release
```

## Installations

1. Install the [MAX SDK](https://docs.modular.com/engine/get-started).

2. git clone this repo and

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
python3 -m pip install -r requirements.txt
```

3. Run the code such as `mojo add.mojo` and `mojo matmul.mojo`. For mnist, first
need to train the model with `python mnist.py` (uses relu) or `python mnist.py --use-relu6`
to train with relu6. Then for inference and checking test accuracy matches the PyTorch version, run
`mojo mnist.mojo` (uses relu) or for relu6 custom op (needs `python mnist.py --use-relu6` to train with relu6),
ensure to run `mojo package custom_ops` and follow with `mojo mnist.mojo --use-relu6`.


Note: `mojo build mnist.mojo` builds the `mnist` binary and we can execute the binary via `./mnist`


4. If you encounter any issues, please make sure to check out the [roadmap and known issues first](https://docs.modular.com/max/roadmap)
