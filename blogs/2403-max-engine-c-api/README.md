# MAX Engine C API Intro

Published blog post: [https://www.modular.com/blog/getting-started-with-max-engine-c-api](https://www.modular.com/blog/getting-started-with-max-engine-c-api)

Versions used in this blog

```txt
max 24.1.0 (c176f84d)
Modular version 24.1.0-c176f84d-release
```

## Installations

1. Install the [MAX SDK](https://docs.modular.com/engine/get-started).

2. git clone and

```sh
python3 -m venv venv && source venv/bin/activate
python3 -m pip install --find-links "$(modular config max.path)/wheels" max-engine
python3 -m pip install -r requirements.txt
```

Finally run end-to-end

```sh
bash run.sh
```
