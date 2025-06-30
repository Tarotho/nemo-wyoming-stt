# NeMo Wyoming STT Server

A local speech-to-text server based on NVIDIA NeMo Conformer models implementing the Wyoming protocol for Home Assistant integration.

## Requirements

- NVIDIA GPU with CUDA support (e.g., Tesla P4)  
- Docker & Docker Compose installed

## Quick Start

'''bash
git clone https://github.com/yourusername/nemo-wyoming-stt.git
cd nemo-wyoming-stt
docker-compose up --build
'''

The server will start listening on port 10300 (Wyoming protocol).

## Home Assistant Configuration

Add this to your `configuration.yaml` to connect HA to the server:

'''yaml
stt:
  - platform: wyoming
    name: nemo_stt
    host: YOUR_SERVER_IP
    port: 10300
'''

## Features

- 100% local speech-to-text inference  
- Uses NeMo Conformer large Spanish model  
- Supports GPU acceleration with NVIDIA CUDA  
- Fully compatible with Home Assistant via Wyoming protocol  
- Dockerized for easy deployment and portability  

## Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and open pull requests.
