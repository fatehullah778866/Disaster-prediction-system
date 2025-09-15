
# Disaster Prediction System

## Overview
The Disaster Prediction System is a comprehensive web application designed to predict and analyze natural disasters such as floods, fires, and landslides. It leverages machine learning models and deep learning techniques to provide accurate predictions and visualizations, aiding in disaster management and response.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features
- **Flood Prediction:** Predicts flood risk using local features and historical data.
- **Fire Prediction:** Forecasts forest fire likelihood based on environmental data.
- **Landslide Detection:** Uses deep learning (UNet) for image-based landslide segmentation.
- **Dashboard:** Centralized dashboard for monitoring predictions and results.
- **Media Management:** Upload and manage images for landslide detection.
- **Results Visualization:** View and download prediction results and segmentation masks.

---

## Project Structure
```
Disaster-prediction-system/
├── db.sqlite3
├── manage.py
├── README.md
├── dashboard/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
│       └── dashboard/
├── fire_prediction/
│   ├── forestfires.csv
│   ├── train_model.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── ml_model/
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   └── templates/
│       └── fire_prediction/
├── flood_prediction/
│   ├── flood_with_pakistan_features.csv
│   ├── train_model.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── ml_model/
│   │   ├── model.pkl
│   │   └── scaler.pkl
│   └── templates/
│       └── flood_prediction/
├── HazardWatch/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── land_slide_detection/
│   ├── forms.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── models/
│   └── templates/
├── media/
│   ├── image_*.png
│   ├── mask_image_*.png
│   ├── results/
│   └── uploads/
├── results/
│   └── mask_image_*.png
└── uploads/
	 └── image_*.png
```

---

## Technologies Used
- **Backend:** Django (Python)
- **Frontend:** HTML, CSS, Django Templates
- **Machine Learning:** scikit-learn, pandas, numpy
- **Deep Learning:** PyTorch (UNet for landslide detection)
- **Database:** SQLite3

---

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtualenv (recommended)

### Steps
1. **Clone the repository:**
	```powershell
	git clone https://github.com/fatehullah778866/Disaster-prediction-system.git
	cd Disaster-prediction-system
	```
2. **Create and activate a virtual environment:**
	```powershell
	python -m venv venv
	venv\Scripts\activate
	```
3. **Install dependencies:**
	```powershell
	pip install -r requirements.txt
	```
	If `requirements.txt` is missing, install manually:
	```powershell
	pip install django pandas numpy scikit-learn torch torchvision
	```
4. **Apply migrations:**
	```powershell
	py manage.py migrate
	```
5. **Run the development server:**
	```powershell
	py manage.py runserver
	```
6. **Access the app:**
	Open your browser and go to `http://127.0.0.1:8000/`

---

## Usage
- **Dashboard:** Main page for accessing disaster prediction modules.
- **Flood Prediction:** Upload relevant data or use provided features to get flood risk analysis.
- **Fire Prediction:** Input environmental parameters or use CSV data for fire risk prediction.
- **Landslide Detection:** Upload satellite or terrain images to get segmentation masks and risk assessment.
- **Results:** Download and view prediction results and processed images.

---

## Model Details
### Flood & Fire Prediction
- **Models:** Trained using scikit-learn (RandomForest, Logistic Regression, etc.)
- **Data:**
  - Flood: `flood_with_pakistan_features.csv`
  - Fire: `forestfires.csv`
- **Artifacts:**
  - Model: `model.pkl`
  - Scaler: `scaler.pkl`

### Landslide Detection
- **Model:** UNet (PyTorch)
- **Weights:** `landslide_unet_full_model.pth`
- **Input:** PNG images
- **Output:** Segmentation mask images

---

## API Endpoints
| Module              | Endpoint                      | Method | Description                       |
|---------------------|------------------------------|--------|-----------------------------------|
| Dashboard           | `/dashboard/`                | GET    | Main dashboard                    |
| Flood Prediction    | `/flood_prediction/`         | GET/POST| Flood prediction form & results   |
| Fire Prediction     | `/fire_prediction/`          | GET/POST| Fire prediction form & results    |
| Landslide Detection | `/land_slide_detection/`     | GET/POST| Image upload & segmentation       |

---

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Create a Pull Request

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
- **Author:** Fateh Ullah
- **GitHub:** [fatehullah778866](https://github.com/fatehullah778866)
- **Email:** fatehullah778866@gmail.com

For issues, suggestions, or contributions, please open an issue or contact via email.

---

## Acknowledgements
- scikit-learn, pandas, numpy, PyTorch
- Django Documentation
- Open source contributors

---

## Screenshots
_Add screenshots of the dashboard, prediction forms, and results here._

---

## Notes
- Ensure all model files (`.pkl`, `.pth`) are present in their respective folders.
- For landslide detection, images should be in PNG format and uploaded via the web interface.
- For production deployment, configure proper settings in `HazardWatch/settings.py` and use a robust database.

---

## FAQ
**Q:** What disasters does this system predict?
**A:** Floods, forest fires, and landslides.

**Q:** Can I use my own data?
**A:** Yes, you can upload your own images for landslide detection and CSV data for flood/fire prediction.

**Q:** How do I retrain the models?
**A:** Use the `train_model.py` scripts in `flood_prediction` and `fire_prediction` folders.

---

## References
- [Django Documentation](https://docs.djangoproject.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/)

---
