# clasify_flowers

Download images from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

Extract to ./data/

APP:

    python -m venv .venv
    
    source .venv/bin/activate
    
    pip install -r requirements.txt
    
    python run.py

DOCKER:

    sudo docker-compose up --build
