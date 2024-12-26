## 1. Download celebA dataset 
```bash
curl -L -o ~/Downloads/celeba-dataset.zip https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
```

## 2. Setting up environment & Install dependencies
- Create a virtual environment and activate
    - Linux/mac
      ```bash
      python3 -m venv .venv
      ```
      ```bash
      source .venv/bin/activate
      ```
    - Windows
      ```bash
      virtualenv .venv
      ```
      ```bash
      cd .venv
      ```
      ```bash
      .\Scripts\activate
      ```
  ```bash
  pip3 install -r requirements.txt
  ```
