# AutoFood

A Django-based web application that recognizes food from images using YOLO models and helps estimate portions and pricing. It includes an admin dashboard, user auth, food menu & categories management, food detection and visualization.

## Features
- Image upload and food detection using YOLO (`ai_models/`)
- Menu and categories with prices and images
- Portion/grams estimation and calculated price per detected item
- Auth (register/login), simple dashboard, and checkout flow
- SQLite by default, easy local setup on Windows

## Tech Stack
- Django
- Python
- Ultralytics YOLO, OpenCV, NumPy, Pillow

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

## Sample Screenshots
Food Recognition
<img width="940" height="561" alt="image" src="https://github.com/user-attachments/assets/d928ac09-7819-4063-9f0d-531f98997315" />

Data Analytics Dashboard
<img width="940" height="535" alt="image" src="https://github.com/user-attachments/assets/88a495b6-8609-4309-8991-2d111ac58007" />

Menu Management
<img width="940" height="560" alt="image" src="https://github.com/user-attachments/assets/e0e4f352-236c-4002-a12f-ced0e2deefe9" />



