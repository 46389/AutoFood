import os
from threading import Lock

# Optional imports for YOLO functionality
try:
	from ultralytics import YOLO  # type: ignore
	YOLO_AVAILABLE = True
except Exception:
	YOLO_AVAILABLE = False
	YOLO = None  # type: ignore


_food_model = None
_plate_model = None
_initialized = False
_init_lock = Lock()


def _resolve_model_path(base_dir: str, relative_path: str) -> str:
	# Models are stored under ai_models at project root
	return os.path.join(base_dir, relative_path)


def initialize_models(base_dir: str | None = None) -> None:
	"""
	Idempotently load YOLO models into memory.
	Safe to call multiple times; only the first call performs initialization.
	"""
	global _initialized, _food_model, _plate_model
	if _initialized:
		return

	with _init_lock:
		if _initialized:
			return

		if not YOLO_AVAILABLE:
			_initialized = True  # Avoid retry loop; inference will surface the issue
			return

		if base_dir is None:
			base_dir = os.getcwd()

		food_model_path = _resolve_model_path(base_dir, os.path.join('ai_models', 'yolo11_cbam_best.pt'))
		plate_model_path = _resolve_model_path(base_dir, os.path.join('ai_models', 'yoloe-11m-seg.pt'))

		# Load food segmentation model
		if os.path.exists(food_model_path):
			_food_model = YOLO(food_model_path, task='segment')
		else:
			_food_model = None

		# Load plate detection/segmentation model
		if os.path.exists(plate_model_path):
			_plate_model = YOLO(plate_model_path)
			try:
				# If supported, restrict to plate class
				_plate_model.set_classes(["plate"])  # type: ignore[attr-defined]
			except Exception:
				# Some model variants may not expose set_classes; continue without it
				pass
		else:
			_plate_model = None

		_initialized = True


def get_food_model():
	return _food_model


def get_plate_model():
	return _plate_model


def is_initialized() -> bool:
	return _initialized


