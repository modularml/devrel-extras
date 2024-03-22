# Semantic search with MAX Engine

Published blog post: [Semantic search with MAX Engine](https://www.modular.com/blog/semantic-search-with-max-engine)

Versions used in this blog

```txt
max 24.1.1 (0ab415f7)
Modular version 24.1.1-0ab415f7-release
```

## Installations

1. Install the [MAX SDK](https://docs.modular.com/engine/get-started).

2. git clone this repo and

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
python3 -m pip install -r requirements.txt
```

## Jupyter notebook

Follow the cells in `bge-embed.ipynb`
