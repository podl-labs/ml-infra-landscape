# ML platforms comparison

Run a simple sentiment detector across popular ML platforms.

## CUDA setup


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