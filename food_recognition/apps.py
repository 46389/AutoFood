from django.apps import AppConfig
from django.conf import settings


def _initialize_yolo_models():
    # Lazy import to avoid import-time side effects before Django setup
    from . import yolo_loader
    # Initialize with project base dir so model paths resolve correctly
    yolo_loader.initialize_models(base_dir=str(settings.BASE_DIR))


class FoodRecognitionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'food_recognition'

    def ready(self):
        # Initialize YOLO models once at app startup; internal guard prevents double-load
        try:
            _initialize_yolo_models()
        except Exception as exc:
            # Avoid crashing startup on optional model load errors; inference will report later
            print(f"[FoodRecognition] YOLO init warning: {exc}")
