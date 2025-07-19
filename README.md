# Smart Parking Space Detector

This project detects whether parking slots are **free** or **occupied** using the CNRPark+EXT dataset.

---

## Folder Structure

- `data/raw/` — Original downloaded dataset files  
- `data/processed/` — Preprocessed images ready for training  
- `data/metadata/` — Dataset metadata files such as CSVs  
- `notebooks/parking_space_detector.ipynb` — Main notebook for data exploration, preprocessing, and model training

---

## Dataset

We use the **CNRPark+EXT** segmented parking space images dataset (patches sized 150x150 px) with labels indicating if a spot is free (0) or occupied (1).

---

## Project Steps

1. Download and organize dataset into `data/raw/`
2. Explore and preprocess data in the notebook
3. Train a simple CNN or fine-tune a model (e.g., ResNet, EfficientNet)
4. Evaluate and visualize results in the notebook

---

## Requirements

Install dependencies from `requirements.txt` using:

```bash
pip install -r requirements.txt
