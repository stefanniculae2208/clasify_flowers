# clasify_flowers

Download images from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html and extract to ./data/102flowers/jpg
Download the segmentation files and extract to ./utils/segmim/


APP:

    python -m venv .venv
    
    source .venv/bin/activate
    
    pip install -r requirements.txt
    
    python run.py

DOCKER:

    sudo docker-compose up --build

DEPENDANCIES:

    python
    python3-tk
    
