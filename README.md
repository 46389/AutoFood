# AutoFood

A Django-based web application that recognizes food from images using YOLO models and helps estimate portions and pricing. It includes an admin dashboard, user auth, food menu & categories management, food detection and visualization.

## Features
- Image upload and food detection using YOLO (`ai_models/`)
- Menu and categories with prices and images
- Portion/grams estimation and calculated price per detected item
- Auth (register/login), simple dashboard, and checkout flow
- SQLite by default, easy local setup on Windows

## Tech Stack
- Django 5.x
- Python 3.10–3.12
- Ultralytics YOLO, PyTorch, OpenCV, NumPy, Pillow

## Repository layout
```
AutoFood/                 # Django project settings
food_recognition/         # Main app (models, views, templates, management commands)
ai_models/                # YOLO weights (.pt)
media/                    # User uploads and predictions
static/                   # Static assets (if collected)
manage.py                 # Django entrypoint
requirements.txt          # Python dependencies
```

## Getting started (Windows PowerShell)
1) Create and activate a virtual environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Ensure model weights exist
- Place YOLO weight files in `ai_models/` (already present: `yolo11_cbam_best.pt`, `yoloe-11m-seg.pt`).

4) Apply migrations and create a superuser
```powershell
python manage.py migrate
python manage.py createsuperuser
```

5) Create categories and menu items (recommended)
```powershell
python manage.py setup_categories
python manage.py setup_menu_items
```

6) Run the development server
```powershell
python manage.py runserver
```
Open `http://127.0.0.1:8000/` in your browser.

## How to use
- Register or log in.
- Navigate to the home/dashboard and upload a food image.
- The app will run detection, estimate grams, and compute price based on the seeded menu.
- Review results and proceed to checkout if needed.

## License
Educational use only. Add an explicit license if you plan to distribute.
