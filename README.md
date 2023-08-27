# ML platforms comparison

We run a simple sentiment detector across various ML platforms and compare the dev experience.

```py
from transformers import pipeline

def get_sentiment(text: str):
    model = pipeline("text-classification")
    return model(text)

def main():
    res = get_sentiment("good job")
    print(res)

if __name__ == "__main__":
    main()
```

Run code locally using

```sh
cd local
poetry shell
python local/main.py
```

## 1. Modal

```sh
cd modal
poetry shell
modal run main.py
```

### Pros

1. **Annotations**: By updating a few lines of code, I was able to turn our python program into a Modal app.

  ```py
  # This function can now be called in the cloud using get_sentiment.remote("gg")
  @stub.function()
  def get_sentiment(text: str):

  # Turns the function into a REST API
  @web_endpoint()
  def foo():
  ```

2. **Infrastructure as code**: Python dependencies are finnicky. I had spent a day trying to downgrade to CUDA 11 on my computer to run tensorflow. The usual way for deterministic builds across different computers is to package your app as a docker image. Docker guarantees determinism by running the app inside a Linux container, howerver this adds a lot of overhead. First you must create the docker image, and running it locally is slower. Fortunately with Modal I could simply define my dependencies in Python. The app runs on the cloud so it runs much faster.

  ```py
  stub = modal.Stub("text-classification")
  transformers_image = modal.Image.debian_slim().pip_install("transformers", "tensorflow")
  ```

3. **Horizontal scalability**: Arguably the most useful feature. We can scale out horizontally in a non-blocking manner and gather the results using `map()`.

  ```py
  get_sentiment.local("gg") # Run locally
  get_sentiment.remote("gg") # Run remote
  ```

4. **Cron and REST API support**: We can write full stack applications and tasks within modal, without leaving Python. Modal calls this cloud 2.0- AWS like features with superior developer experience.

5. **Free credits**: $100 credits to get started is nifty.

6. **Excellent examples**: Modal has a done a good job having lot of examples related to AI/ML. For example generating images on stable diffusion, scraping etc.

### Cons

1. **Vendor lock in**: The resultant program runs only on Modal. Notice how `transformers` and `tensorflow` are no longer imported at the top? I can't run the app locally now.

```py
import modal

stub = modal.Stub("text-classification")
transformers_image = modal.Image.debian_slim().pip_install("transformers", "tensorflow")

@stub.function(image=transformers_image)
def get_sentiment(text: str):
    from transformers import pipeline

    model = pipeline("text-classification")
    return model(text)
```

2. **Integration overhead**: What if I wish to integrate modal with my existing server running on AWS EC2? I cannot simply expose a REST API and call it without compromising security. I'll have to build a custom API key protected API.

3. It took longer to build and run that what is claimed on Modal's site. `time modal run modal/main.py` gave the results as follows. Still impressive but short of the claimed numbers.

```
real    0m17.309s
user    0m1.469s
sys     0m0.142s
```

## 2. Beam.cloud

Next, I gave beam.cloud a try. First impression- Modal's blue twin. Notice how the API resembles Modal. They even provide identical functionality for cron jobs and REST APIs.

```sh
from beam import App, Runtime, Image
from transformers import pipeline

app = App(
    name="text-classification",
    runtime=Runtime(
        image=Image(
            python_version="python3.10",
            python_packages=["transformers", "tensorflow"],
        ),
    ),
)


def get_sentiment(text: str):
    model = pipeline("text-classification")
    return model(text)

@app.run()
def main():
    res = get_sentiment("good job")
    print(res)

```


Try out the code with

```sh
# Create a run.cloud account. Follow their instructions to setup beam.cloud CLI on your system.

# Run code
cd beam
poetry shell
beam run main.py:main
```

### Pros

1. All the features discussed for Modal- infra as code, free credits, etc.
2. Good quality examples
3. This is subjective but I find the CLI easier to use. I can define a bunch of functions and call the one I need with `beam run main.py:main` or `beam run main.py:get_sentiment`

### Cons

1. Supports horizontal scaling of APIs but there's no equivalent to `get_sentiment.map()`.
2. Pricing is slightly higher than Modal.

## 3. Baseten

Unlike the others [Baseten](https://baseten.co) intends on doing one thing right, "machine learning infra that just works". No cron jobs or annotations, Baseten focuses on quickly getting containers up. They've built an open source tool [truss]() to quickly package and deploy models. Our code becomes like this

```py
from transformers import pipeline

class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self.model = pipeline("text-classification")

    def predict(self, model_input: str):
        return self.model(model_input)
```

To build and run a truss:

```sh
# Install truss globally
pip install truss

# create a project called baseten
truss init baseten

# deploy
truss push

# run
truss predict -d '"good job"'

curl -X POST https://app.baseten.co/models/{MODEL_ID}/predict \
  -H 'Authorization: Api-Key {API_KEY}' \
  -d '"good game"'

# run locally
truss image run

curl -X POST http://127.0.0.1:8080/v1/models/model:predict \
    -d '"good game"'

# run locally with docker
truss image build
docker run -p 8080:8080 custom-model
```

### How does truss work?

Truss installs specified dependencies and puts your model behind a server called [inference_server.py](https://github.com/basetenlabs/truss/blob/3e9c1c5090d88182a3d8257fd03f26d435baec24/truss/templates/server/inference_server.py). The function `predict` is exposed on port 8080.

```py
import os
from typing import Dict

import yaml
from common.truss_server import TrussServer  # noqa: E402
from shared.logging import setup_logging

CONFIG_FILE = "config.yaml"

setup_logging()


class ConfiguredTrussServer:
    _config: Dict
    _port: int

    def __init__(self, config_path: str, port: int):
        self._port = port
        with open(config_path, encoding="utf-8") as config_file:
            self._config = yaml.safe_load(config_file)

    def start(self):
        server = TrussServer(http_port=self._port, config=self._config)
        server.start()


if __name__ == "__main__":
    env_port = int(os.environ.get("INFERENCE_SERVER_PORT", "8080"))
    ConfiguredTrussServer(CONFIG_FILE, env_port).start()
```

### Pros

1. **Open source**: Truss is an open source tool to wrap models into a docker container. Besides being used for deploying to baseten, you can also use it standalone to run containers locally.
2. **Fast builds**: Truss is a boon for local development but local docker builds are slow. It took me 20 minutes to get a simple text classification container ready. Baseten deployed it in a minute.
3. **One click deployments**: Baseten maintains a truss library for popular models like stable diffusion and llama. You can deploy them in a single click from the baseten console.
4. **No code application builder**: Baseten provides a no-code application builder that seemlessly integrates with the ML backend.

### Cons

1. **Single purpose**: Baseten is a backend tool. It has no functionality such as REST APIs for clients. Instead you must call baseten from a secure server which is responsible for the frontend. On the flip side this means baseten is good if you intend to integrate with your existing server stack. You can also use your models with their application builder.
2. **Closed beta**: Your account must be whitelisted in order to use baseten.

## Local environment setup

1. Setup keyring, install CUDA. Ref- https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get -y install cuda-11-8 # just cuda will install v12 which isn't well supported
```

2. Setup env vars in ~/.bashrc

```bashrc
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

3. Reboot

4. Run `nvcc --version`

5. Install cudnn

```sh
sudo apt install -y libcudnn8 libcudnn8-dev libcudnn8-samples
```


Remove everything


```sh
# Remove CUDA and Nvidia
sudo apt remove nvidia-driver-525  nvidia-dkms-525
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

sudo dpkg -r cuda-repo-ubuntu2204-11-8-local
sudo dpkg --purge cuda-repo-ubuntu2204-11-8-local

sudo dpkg -r --force-all cuda-repo-ubuntu2204-12-2-local
sudo dpkg --purge cuda-repo-ubuntu2204-12-2-local

# Remove old CUDA
sudo apt remove cuda libcudnn8 libcudnn8-dev libcudnn8-samples nvidia-cuda-toolkit
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo rm -rf /usr/local/cuda*

# Remove keyring
sudo dpkg -r cuda-keyring
sudo dpkg --purge cuda-keyring
```

## Python version management

```sh
# Install pyenv
curl https://pyenv.run | bash

# Install version
pyenv install 3.10

pyenv global 3.10

pyenv local 3.10

poetry env use 3.10
```