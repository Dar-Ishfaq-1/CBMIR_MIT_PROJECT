
# Content-Based Medical Image Retrieval (CBMIR)

This repository contains a Master’s level academic project on Content-Based Medical Image Retrieval (CBMIR).  
The system retrieves visually similar medical images using deep learning–based feature representations and similarity search.

---

## Project Overview
The objective of this project is to design a CBMIR system that retrieves similar medical images based on their visual content. The project combines deep feature extraction, efficient indexing, and explainability techniques to make the retrieval process more interpretable.

---

## Workflow
Input Image → Feature Extraction → Feature Indexing → Similarity Search → Retrieved Images → Explainability (XAI)

---

## Main Components
- Feature extraction using Vision Transformer (ViT)
- Similarity search using FAISS
- Query-based medical image retrieval
- Explainable AI (XAI) for visual interpretation of retrieved results
- Python-based implementation

---

## Explainable AI (XAI)
Explainable AI techniques are used to provide visual insights into the image retrieval process. XAI helps in understanding which regions of the input image contribute most to feature extraction and similarity matching, improving transparency and interpretability of the system.

---

## Project Structure
```

CBMIR-System/
│
├── src/
│   └── Int_project.py
│
├── notebooks/
│   └── Project_Interface_file.ipynb
│
├── config/
│   └── vit_metadata.json
│
├── README.md
├── requirements.txt
└── .gitignore

````

---

## Technologies Used
- Python
- PyTorch
- Vision Transformer (ViT)
- FAISS
- NumPy
- OpenCV
- Explainable AI (XAI)

---

## Installation
```bash
pip install -r requirements.txt
````

---

## How to Run

```bash
python src/Int_project.py
```

---

## Notes

* Model weight files (`.pth`), FAISS index files (`.index`), datasets, and images are not included in this repository due to size limitations.
* These files can be generated locally by running the feature extraction and indexing steps.

---

## Team Members

* Ishfaq Ahmad Dar
* Midhat Un Nisa
* Aasif Ahmad Pala

---

## Academic Use

## License
This project is licensed under the MIT License.
