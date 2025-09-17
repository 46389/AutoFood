from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json as json_module
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from django.db.models import Sum, Count, Avg
from django.db import models
from django.utils import timezone
import json
import os
import io
import base64

# Optional imports for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Optional imports for YOLO functionality
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .models import Detection, DetectedItem, MenuItem, FoodCategory
from .forms import ImageUploadForm, MenuItemForm, DetectedItemForm, CustomUserCreationForm, FoodCategoryForm, MenuSearchForm

class CustomLoginView(LoginView):
    template_name = 'registration/login.html'
    redirect_authenticated_user = True
    
    def form_invalid(self, form):
        """Add custom error message for invalid login"""
        messages.error(self.request, 'Invalid username or password. Please try again.')
        return super().form_invalid(form)

def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, 'Registration successful! Please log in.')
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def home_view(request):
    form = ImageUploadForm()
    context = {
        'form': form,
        'menu_items': MenuItem.objects.all()
    }
    return render(request, 'food_recognition/home.html', context)

@login_required
def process_image(request):
    if request.method == 'POST':
        # Check if this is a reprocessing request
        reprocess_id = request.POST.get('reprocess')
        if reprocess_id:
            try:
                detection = get_object_or_404(Detection, id=reprocess_id, user=request.user)
                confidence_threshold = float(request.POST.get('confidence_threshold', 0.5))
                
                # Clear existing detection items
                detection.items.all().delete()
                
                # Clear existing segmented image
                if detection.segmented_image:
                    detection.segmented_image.delete()
                
                # Update confidence threshold
                detection.confidence_threshold = confidence_threshold
                detection.save()
                
                image_path = detection.original_image.path
            except (ValueError, Detection.DoesNotExist):
                messages.error(request, 'Invalid reprocessing request.')
                return redirect('home')
        else:
            # Regular image upload
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                image_file = form.cleaned_data['image']
                confidence_threshold = form.cleaned_data['confidence_threshold']
                
                # Save the uploaded image
                detection = Detection.objects.create(
                    user=request.user,
                    original_image=image_file,
                    confidence_threshold=confidence_threshold
                )
                image_path = detection.original_image.path
            else:
                messages.error(request, 'Please provide a valid image.')
                return redirect('home')
        
        # Run YOLO inference (for both new uploads and reprocessing)
        try:
            results, onnx_failed = run_yolo_inference(image_path, confidence_threshold)
            
            # Calculate food portions relative to plates
            food_results, estimated_portions = calculate_food_portions(results)
            
            # Save detection results (exclude plates from detected items)
            for result in food_results:
                estimated_grams = estimated_portions.get(result.get('id'), 0)
                
                # Find matching menu item to check pricing unit
                matching_menu_item = None
                try:
                    matching_menu_item = MenuItem.objects.filter(
                        name__icontains=result['class']
                    ).first()
                except:
                    pass
                
                # Determine quantity based on pricing unit
                if matching_menu_item and matching_menu_item.pricing_unit == 'serving':
                    estimated_servings = estimated_grams / 100.0  # 1 serving = 100g
                    quantity = max(0.01, round(estimated_servings, 2))  # Round to 2 decimal places, minimum 0.01 serving
                else:
                    # For per item pricing, use integer quantity (assume 1 item detected)
                    quantity = 1.0
                
                # Calculate item price
                item_price = 0
                if matching_menu_item:
                    item_price = float(matching_menu_item.price) * quantity
                
                DetectedItem.objects.create(
                    detection=detection,
                    class_label=result['class'],
                    confidence=result['confidence'],
                    quantity=quantity,
                    bbox_x=result['bbox'][0],
                    bbox_y=result['bbox'][1],
                    bbox_width=result['bbox'][2],
                    bbox_height=result['bbox'][3],
                    mask_data=json.dumps(result.get('mask', {})),
                    estimated_grams=estimated_grams,
                    calculated_price=item_price,
                    menu_item=matching_menu_item
                )
            
            # Generate segmented image with bounding boxes and masks
            try:
                segmented_image_path = generate_segmented_image(image_path, results)
                if segmented_image_path:
                    with open(segmented_image_path, 'rb') as f:
                        detection.segmented_image.save(
                            f'segmented_{detection.id}.jpg',
                            ContentFile(f.read())
                        )
                    # Clean up temporary file
                    os.remove(segmented_image_path)
            except Exception as e:
                print(f"Error generating segmented image: {e}")
            
        except Exception as e:
            # Handle YOLO inference errors
            error_message = str(e)
            if "No food detected" in error_message:
                messages.warning(request, 'No food detected in the image. Please try with a different image or adjust the confidence threshold.')
            else:
                messages.error(request, f'Error processing image: {error_message}')
            
            # Delete the detection record if inference failed (only for new uploads, not reprocessing)
            if not reprocess_id:
                detection.delete()
            return redirect('home')
        
        return redirect('detection_results', detection_id=detection.id)
    
    return redirect('home')

def run_yolo_inference(image_path, confidence_threshold):
    """
    Run YOLOv11-seg inference on the image for food detection
    and YOLO-World for plate detection
    """
    if not YOLO_AVAILABLE:
        raise Exception("YOLO dependencies not available. Please install required packages: pip install ultralytics opencv-python")
    
    try:
        detections = []
        
        # 1. Food detection with YOLOv11-seg - use PT file
        pt_model_path = os.path.join('ai_models', 'yolo11_cbam_best.pt')
        
        onnx_failed = True  # Always report as ONNX failed since we're using PT
        
        if os.path.exists(pt_model_path):
            food_model = YOLO(pt_model_path, task='segment')
            food_results = food_model(
                image_path, 
                conf=confidence_threshold, 
                iou=0.45,  # Default YOLO IoU threshold
                max_det=300,  # Maximum detections
                agnostic_nms=False,
                verbose=False
            )
        else:
            raise Exception("PT model file not found for food detection")
        
        
        for result in food_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Handle masks if available (segmentation model)
                masks = None
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    print(f"Found food masks with shape: {masks.shape}")
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    mask_data = None
                    if masks is not None and i < len(masks):
                        # Get the original image dimensions for proper mask sizing
                        original_shape = result.orig_shape  # (height, width)
                        mask = masks[i]
                        
                        # Resize mask to original image dimensions if needed
                        # mask.shape is (H, W), original_shape is (H, W)
                        # cv2.resize expects (width, height) format
                        if mask.shape != (original_shape[0], original_shape[1]):
                            import cv2
                            mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_resized = mask
                        
                        # Store mask as numpy array info for later processing
                        mask_data = {
                            'shape': mask_resized.shape,
                            'data': mask_resized.tolist(),
                            'original_shape': original_shape
                        }
                        print(f"Food mask {i} original shape: {mask.shape}, resized to: {mask_resized.shape}")
                    
                    detections.append({
                        'id': f"food_{len(detections)}",
                        'class': result.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': [float(x) for x in box],
                        'mask': mask_data,
                        'type': 'food'
                    })
        
        # 2. Plate detection with YOLOE - use PT file
        try:
            # Initialize YOLOE model for plate detection
            plate_pt_path = os.path.join('ai_models', 'yoloe-11m-seg.pt')
            
            if os.path.exists(plate_pt_path):
                plate_model = YOLO(plate_pt_path)
                print(f"Loaded plate PT model as segmentation model")
                
                # Set text prompt for YOLOE to detect plates
                plate_model.set_classes(["plate"])
            else:
                raise Exception("PT model file not found for plate detection")
            
            # Run plate detection with text prompt
            plate_results = plate_model(
                image_path, 
                conf=confidence_threshold,
                iou=0.45,  # Default YOLO IoU threshold
                max_det=300,  # Maximum detections
                agnostic_nms=False,
                verbose=False
            )
            
            for result in plate_results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Handle masks if available (segmentation model only)
                    masks = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        print(f"Found plate masks with shape: {masks.shape}")
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        mask_data = None
                        if masks is not None and i < len(masks):
                            # Get the original image dimensions for proper mask sizing
                            original_shape = result.orig_shape  # (height, width)
                            mask = masks[i]
                            
                            # Resize mask to original image dimensions if needed
                            # mask.shape is (H, W), original_shape is (H, W)
                            # cv2.resize expects (width, height) format
                            if mask.shape != (original_shape[0], original_shape[1]):
                                import cv2
                                mask_resized = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_resized = mask
                            
                            # Store mask as numpy array info for later processing
                            mask_data = {
                                'shape': mask_resized.shape,
                                'data': mask_resized.tolist(),
                                'original_shape': original_shape
                            }
                            print(f"Plate mask {i} original shape: {mask.shape}, resized to: {mask_resized.shape}")
                        else:
                            print(f"No mask available for plate detection {i} (detection-only model)")
                        
                        detections.append({
                            'id': f"plate_{len(detections)}",
                            'class': 'plate',  # Use the text prompt class
                            'confidence': float(conf),
                            'bbox': [float(x) for x in box],
                            'mask': mask_data,
                            'type': 'plate'
                        })
            
            print(f"Detected {len([d for d in detections if d['type'] == 'plate'])} plates/bowls")
                        
        except Exception as e:
            print(f"Plate detection failed (continuing with food detection only): {e}")
        
        # Check if no objects were detected
        if len(detections) == 0:
            raise Exception("No food or plates detected in the image")
        
        food_count = len([d for d in detections if d['type'] == 'food'])
        plate_count = len([d for d in detections if d['type'] == 'plate'])
        
        print(f"Total detections (PT): {len(detections)} (Food: {food_count}, Plates: {plate_count})")
        
        # Debug: Print all detected classes
        for detection in detections:
            print(f"  - {detection['type']}: {detection['class']} (conf: {detection['confidence']:.3f})")
        
        return detections, onnx_failed
        
    except FileNotFoundError:
        raise Exception("YOLO model file not found. Please ensure 'yolo11_cbam_best.pt' and 'yoloe-11m-seg.pt' model file exists in the project directory.")
    except Exception as e:
        # Re-raise the exception to be handled by the caller
        raise Exception(f"YOLO inference failed: {str(e)}")

def calculate_food_portions(detection_results):
    """
    Calculate food portions relative to each plate
    Returns: (food_results, estimated_portions_dict)
    """
    import math
    
    # Food density estimates (g/cm³)
    FOOD_DENSITIES = {
        'rice': 0.8,
        'noodles': 0.7,
        'chicken': 1.0,
        'beef': 1.0,
        'fish': 1.0,
        'vegetables': 0.6,
        'soup': 1.0,
        'default': 0.8
    }
    
    # Separate food and plates
    food_items = [d for d in detection_results if d['type'] == 'food']
    plate_items = [d for d in detection_results if d['type'] == 'plate']
    
    estimated_portions = {}
    
    # Remove duplicate rice detections per plate
    food_items = remove_duplicate_rice_per_plate(food_items, plate_items)
    
    # If no plates detected, use default portion estimation
    if not plate_items:
        for food in food_items:
            bbox = food['bbox']
            food_area_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # Assume medium portion without plate reference
            estimated_grams = max(25, min(200, food_area_pixels * 0.01))
            estimated_portions[food['id']] = estimated_grams
        return food_items, estimated_portions
    
    # Calculate portions for each food item relative to its nearest plate
    for food in food_items:
        # Find the nearest plate to this food item
        nearest_plate = find_nearest_plate(food, plate_items)
        if not nearest_plate:
            # Fallback to default calculation
            bbox = food['bbox']
            food_area_pixels = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            estimated_grams = max(25, min(200, food_area_pixels * 0.01))
            estimated_portions[food['id']] = estimated_grams
            continue
            
        plate_bbox = nearest_plate['bbox']
        
        # Calculate plate dimensions in pixels
        plate_width_pixels = plate_bbox[2] - plate_bbox[0]
        plate_height_pixels = plate_bbox[3] - plate_bbox[1]
        plate_diameter_pixels = (plate_width_pixels + plate_height_pixels) / 2
        
        # Assume plate diameter is 23cm
        PLATE_DIAMETER_CM = 23
        pixels_per_cm = plate_diameter_pixels / PLATE_DIAMETER_CM
        
        bbox = food['bbox']
        food_width_pixels = bbox[2] - bbox[0]
        food_height_pixels = bbox[3] - bbox[1]
        food_area_pixels = food_width_pixels * food_height_pixels
        
        # Convert to cm²
        food_area_cm2 = food_area_pixels / (pixels_per_cm ** 2)
        
        # Estimate height (assume 1-3cm depending on food type)
        food_type = food['class'].lower()
        if any(grain in food_type for grain in ['rice', 'noodles', 'pasta']):
            estimated_height_cm = 2.0
        elif any(meat in food_type for meat in ['chicken', 'beef', 'fish', 'meat']):
            estimated_height_cm = 2.5
        elif 'soup' in food_type:
            estimated_height_cm = 3.0
        else:
            estimated_height_cm = 1.5
        
        # Calculate volume
        food_volume_cm3 = food_area_cm2 * estimated_height_cm
        
        # Get density and calculate weight
        density = FOOD_DENSITIES.get(food_type, FOOD_DENSITIES['default'])
        estimated_grams = food_volume_cm3 * density
        
        # Apply reasonable bounds (10g to 500g per food item)
        estimated_grams = max(10, min(500, estimated_grams))
        
        estimated_portions[food['id']] = estimated_grams
        
        print(f"Food: {food['class']}, Plate: {nearest_plate['id']}, Area: {food_area_cm2:.1f}cm², "
              f"Volume: {food_volume_cm3:.1f}cm³, Weight: {estimated_grams:.0f}g")
    
    return food_items, estimated_portions

def find_nearest_plate(food_item, plate_items):
    """
    Find the nearest plate to a food item based on center distances
    """
    if not plate_items:
        return None
        
    food_bbox = food_item['bbox']
    food_center_x = (food_bbox[0] + food_bbox[2]) / 2
    food_center_y = (food_bbox[1] + food_bbox[3]) / 2
    
    min_distance = float('inf')
    nearest_plate = None
    
    for plate in plate_items:
        plate_bbox = plate['bbox']
        plate_center_x = (plate_bbox[0] + plate_bbox[2]) / 2
        plate_center_y = (plate_bbox[1] + plate_bbox[3]) / 2
        
        # Calculate Euclidean distance
        distance = ((food_center_x - plate_center_x) ** 2 + (food_center_y - plate_center_y) ** 2) ** 0.5
        
        # Check if food is within or overlapping with plate
        food_overlaps_plate = (
            food_bbox[0] < plate_bbox[2] and food_bbox[2] > plate_bbox[0] and
            food_bbox[1] < plate_bbox[3] and food_bbox[3] > plate_bbox[1]
        )
        
        if food_overlaps_plate:
            # If food overlaps with plate, prioritize this plate
            distance = distance * 0.1  # Give heavy preference to overlapping plates
        
        if distance < min_distance:
            min_distance = distance
            nearest_plate = plate
    
    return nearest_plate

def remove_duplicate_rice_per_plate(food_items, plate_items):
    """
    Remove duplicate rice detections within each plate, keeping only the one with highest confidence
    """
    if not plate_items:
        # If no plates, remove duplicate rice globally
        rice_items = [item for item in food_items if 'rice' in item['class'].lower()]
        if len(rice_items) > 1:
            # Keep only the rice with highest confidence
            best_rice = max(rice_items, key=lambda x: x['confidence'])
            food_items = [item for item in food_items if item == best_rice or 'rice' not in item['class'].lower()]
        return food_items
    
    processed_food_items = []
    
    # Group food items by their nearest plate
    plate_food_groups = {}
    for food in food_items:
        nearest_plate = find_nearest_plate(food, plate_items)
        if nearest_plate:
            plate_id = nearest_plate['id']
            if plate_id not in plate_food_groups:
                plate_food_groups[plate_id] = []
            plate_food_groups[plate_id].append(food)
        else:
            # Food items without nearby plates are processed separately
            processed_food_items.append(food)
    
    # Process each plate's food items
    for plate_id, foods_in_plate in plate_food_groups.items():
        # Find all rice items in this plate
        rice_items = [item for item in foods_in_plate if 'rice' in item['class'].lower()]
        non_rice_items = [item for item in foods_in_plate if 'rice' not in item['class'].lower()]
        
        # Keep only the best rice item per plate
        if rice_items:
            best_rice = max(rice_items, key=lambda x: x['confidence'])
            processed_food_items.extend(non_rice_items + [best_rice])
        else:
            processed_food_items.extend(non_rice_items)
    
    # Handle orphaned rice items (not near any plate)
    orphaned_rice = [item for item in processed_food_items if 'rice' in item['class'].lower() and not any(find_nearest_plate(item, plate_items) for item in processed_food_items if 'rice' in item['class'].lower())]
    if len(orphaned_rice) > 1:
        best_orphaned_rice = max(orphaned_rice, key=lambda x: x['confidence'])
        processed_food_items = [item for item in processed_food_items if item == best_orphaned_rice or 'rice' not in item['class'].lower()]
    
    print(f"Removed duplicate rice: Original {len(food_items)} items -> {len(processed_food_items)} items")
    return processed_food_items

def generate_segmented_image(image_path, detection_results):
    """
    Generate an image with bounding boxes and segmentation masks overlaid
    """
    if not YOLO_AVAILABLE:
        return None
    
    try:
        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_display = img.copy()
        
        # Create an overlay for masks
        mask_overlay = img.copy()
        
        # Define colors for different types (BGR format)
        food_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        plate_colors = [
            (128, 128, 128),  # Gray
            (64, 64, 64),     # Dark gray
            (192, 192, 192),  # Light gray
        ]
        
        food_count = 0
        plate_count = 0
        
        for i, result in enumerate(detection_results):
            detection_type = result.get('type', 'food')
            
            # Choose color based on detection type
            if detection_type == 'plate':
                color = plate_colors[plate_count % len(plate_colors)]
                plate_count += 1
            else:
                color = food_colors[food_count % len(food_colors)]
                food_count += 1
            
            bbox = result['bbox']
            class_name = result['class']
            confidence = result['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label and confidence with type indicator
            type_prefix = "[P]" if detection_type == 'plate' else "[F]"
            label = f"{type_prefix} {class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img_display, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_display, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw segmentation mask if available
            if result.get('mask') and result['mask'] is not None:
                try:
                    mask_info = result['mask']
                    if isinstance(mask_info, dict) and 'data' in mask_info:
                        # Reconstruct the mask from stored data
                        mask_data = np.array(mask_info['data'])
                        mask_shape = mask_info['shape']
                        original_shape = mask_info.get('original_shape', mask_shape)
                        
                        print(f"Processing mask for {class_name} with shape: {mask_shape}")
                        print(f"Image shape: {img.shape}")
                        
                        # Ensure mask is in the right format
                        if len(mask_data.shape) == 1:
                            # Reshape to stored dimensions
                            mask = mask_data.reshape(mask_shape)
                        else:
                            mask = mask_data
                        
                        # Convert to proper data type and scale
                        if mask.dtype != np.uint8:
                            mask = (mask * 255).astype(np.uint8)
                        
                        # Resize mask to match image dimensions if needed
                        if mask.shape != img.shape[:2]:
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                            print(f"Final mask resized to: {mask.shape}")
                        
                        # Create colored mask overlay
                        colored_mask = np.zeros_like(img)
                        
                        # Apply color where mask is present (threshold at 127 for uint8)
                        mask_bool = mask > 127
                        colored_mask[mask_bool] = color
                        
                        # Blend with more opacity for visibility
                        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.8, 0)
                        
                        print(f"Applied mask for {class_name}, mask pixels: {np.sum(mask_bool)}")
                        
                except Exception as e:
                    print(f"Error processing mask for {class_name}: {e}")
                    print(f"Mask data type: {type(result.get('mask'))}")
                    if result.get('mask'):
                        print(f"Mask content: {result['mask']}")
            else:
                # If no mask is available, create a semi-transparent rectangle overlay
                print(f"No mask available for {class_name}, using bounding box overlay")
                rect_overlay = np.zeros_like(img)
                cv2.rectangle(rect_overlay, (x1, y1), (x2, y2), color, -1)
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, rect_overlay, 0.5, 0)
        
        # Combine the image with mask overlay for better visibility
        final_image = cv2.addWeighted(img_display, 0.5, mask_overlay, 0.5, 0)
        
        # Save the segmented image to a temporary file
        temp_path = f"temp_segmented_{timezone.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_path, final_image)
        
        return temp_path
        
    except Exception as e:
        print(f"Error in generate_segmented_image: {e}")
        return None

@login_required
def detection_results(request, detection_id):
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    items = detection.items.all()
    menu_items = MenuItem.objects.all()
    
    # Calculate total price
    total_price = detection.get_calculated_total_price()
    
    context = {
        'detection': detection,
        'items': items,
        'menu_items': menu_items,
        'total_price': total_price,
    }
    return render(request, 'food_recognition/results.html', context)

@login_required
def edit_detection(request, item_id):
    item = get_object_or_404(DetectedItem, id=item_id, detection__user=request.user)
    
    if request.method == 'POST':
        class_label = request.POST.get('class_label', '').strip()
        quantity = float(request.POST.get('quantity', 1.0))
        menu_item_id = request.POST.get('menu_item')
        
        try:
            # Update basic fields
            item.class_label = class_label

            # Update menu item if provided
            if menu_item_id:
                menu_item = MenuItem.objects.get(id=menu_item_id)
                item.menu_item = menu_item

                # Set quantity based on pricing unit
                if menu_item.pricing_unit == 'item':
                    # For per item pricing, use integer quantity (minimum 1)
                    item.quantity = max(1, int(round(quantity)))
                else:
                    # For per serving pricing, allow decimal quantity (minimum 0.01)
                    item.quantity = max(0.01, round(quantity, 2))

                # Recalculate price
                item.calculated_price = float(menu_item.price) * item.quantity
            else:
                item.menu_item = None
                item.quantity = max(0.01, round(quantity, 2))  # Default to decimal when no menu item
                item.calculated_price = 0.00
            
            item.save()
            
            messages.success(request, f'Updated "{class_label}" (Qty: {item.quantity:.2f}) successfully!')
            return redirect('detection_results', detection_id=item.detection.id)
            
        except (MenuItem.DoesNotExist, ValueError) as e:
            messages.error(request, 'Invalid menu item or quantity. Please try again.')
            return redirect('detection_results', detection_id=item.detection.id)
    
    # For GET requests, redirect back to results (shouldn't happen with modal)
    return redirect('detection_results', detection_id=item.detection.id)

@login_required
def delete_detection(request, item_id):
    item = get_object_or_404(DetectedItem, id=item_id, detection__user=request.user)
    detection_id = item.detection.id
    item.delete()
    messages.success(request, 'Item deleted successfully!')
    return redirect('detection_results', detection_id=detection_id)

@login_required
def add_manual_item(request, detection_id):
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    
    if request.method == 'POST':
        menu_item_id = request.POST.get('menu_item')
        quantity = float(request.POST.get('quantity', 1.0))
        
        try:
            menu_item = MenuItem.objects.get(id=menu_item_id)

            # Set quantity based on pricing unit
            if menu_item.pricing_unit == 'item':
                # For per item pricing, use integer quantity (minimum 1)
                final_quantity = max(1, int(round(quantity)))
            else:
                # For per serving pricing, allow decimal quantity (minimum 0.01)
                final_quantity = max(0.01, round(quantity, 2))

            # Calculate price
            calculated_price = float(menu_item.price) * final_quantity

            # Create the detected item
            DetectedItem.objects.create(
                detection=detection,
                menu_item=menu_item,
                class_label=menu_item.name,  # Use menu item name as class label
                confidence=1.0,
                quantity=final_quantity,
                bbox_x=0,
                bbox_y=0,
                bbox_width=100,
                bbox_height=100,
                estimated_grams=0.0,  # Manual items don't have weight estimation
                calculated_price=calculated_price,
                is_manually_added=True
            )
            
            messages.success(request, f'Added {menu_item.name} (Qty: {final_quantity:.2f}) successfully!')
            return redirect('detection_results', detection_id=detection_id)
            
        except (MenuItem.DoesNotExist, ValueError) as e:
            messages.error(request, 'Invalid menu item or quantity. Please try again.')
            return redirect('detection_results', detection_id=detection_id)
    
    # For GET requests, redirect back to results (shouldn't happen with modal)
    return redirect('detection_results', detection_id=detection_id)

@login_required
@csrf_exempt
def update_item_quantity(request):
    if request.method == 'POST':
        try:
            data = json_module.loads(request.body)
            item_id = data.get('item_id')
            quantity = float(data.get('quantity', 0))
            
            # Get the detected item
            item = get_object_or_404(DetectedItem, id=item_id, detection__user=request.user)

            # Update quantity based on pricing unit
            if item.menu_item and item.menu_item.pricing_unit == 'item':
                # For per item pricing, use integer quantity (minimum 1)
                item.quantity = max(1, int(round(quantity)))
            else:
                # For per serving pricing or no menu item, allow decimal quantity (minimum 0.01)
                item.quantity = max(0.01, round(quantity, 2))

            # Recalculate price
            if item.menu_item:
                item.calculated_price = float(item.menu_item.price) * item.quantity
            
            item.save()
            
            return JsonResponse({
                'success': True,
                'new_quantity': float(item.quantity),
                'new_price': float(item.calculated_price),
                'unit_price': float(item.menu_item.price) if item.menu_item else 0
            })
            
        except (DetectedItem.DoesNotExist, ValueError, json_module.JSONDecodeError) as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def checkout(request, detection_id):
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    
    if request.method == 'POST':
        # Calculate total
        total = 0
        for item in detection.items.all():
            if item.menu_item:
                # Use quantity as servings if pricing unit is serving, otherwise as items
                if item.menu_item.pricing_unit == 'serving':
                    total += float(item.menu_item.price) * float(item.quantity)  # quantity is already in servings
                else:
                    total += float(item.menu_item.price) * max(1, int(item.quantity))  # round up to nearest item
        
        detection.total_amount = total
        detection.is_checked_out = True
        detection.checkout_date = timezone.now()
        detection.save()
        
        messages.success(request, f'Checkout completed! Total: RM{total:.2f}')
        return redirect('dashboard')
    
    return render(request, 'food_recognition/checkout.html', {'detection': detection})

@login_required
def reset_detection(request, detection_id):
    detection = get_object_or_404(Detection, id=detection_id, user=request.user)
    detection.delete()
    messages.info(request, 'Detection cleared successfully!')
    return redirect('home')

@login_required
def dashboard_view(request):
    from datetime import datetime, date, timedelta
    from django.db.models import Q
    
    user_detections = Detection.objects.filter(user=request.user).order_by('-created_at')
    
    # Statistics
    total_detections = user_detections.count()
    completed_checkouts = user_detections.filter(is_checked_out=True).count()
    total_revenue = user_detections.filter(is_checked_out=True).aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    # Today's statistics
    today = date.today()
    today_detections_qs = user_detections.filter(created_at__date=today)
    today_detections = today_detections_qs.count()
    today_revenue = today_detections_qs.filter(is_checked_out=True).aggregate(
        total=Sum('total_amount')
    )['total'] or 0
    
    # Daily revenue data for the last 14 days
    daily_revenue_data = []
    for i in range(13, -1, -1):  # Last 14 days
        target_date = today - timedelta(days=i)
        daily_revenue = user_detections.filter(
            created_at__date=target_date,
            is_checked_out=True
        ).aggregate(total=Sum('total_amount'))['total'] or 0
        
        daily_revenue_data.append({
            'date': target_date.strftime('%Y-%m-%d'),
            'date_display': target_date.strftime('%m/%d'),
            'revenue': float(daily_revenue)
        })
    
    # Popular items from checked out orders
    popular_items = DetectedItem.objects.filter(
        detection__user=request.user,
        detection__is_checked_out=True,
        menu_item__isnull=False
    ).values(
        'menu_item__name', 
        'menu_item__price'
    ).annotate(
        total_quantity=Sum('quantity'),
        order_count=Count('detection', distinct=True),
        total_revenue=Sum('calculated_price')
    ).order_by('-order_count')
    
    context = {
        'detections': user_detections[:10],  # Last 10 detections
        'total_detections': total_detections,
        'completed_checkouts': completed_checkouts,
        'total_revenue': total_revenue,
        'today_detections': today_detections,
        'today_revenue': today_revenue,
        'popular_items': popular_items,
        'daily_revenue_data': daily_revenue_data,
    }
    return render(request, 'food_recognition/dashboard.html', context)

@login_required
def export_dashboard_data(request):
    import csv
    from datetime import date
    from django.http import HttpResponse
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="dashboard_export_{date.today()}.csv"'
    
    writer = csv.writer(response)
    
    # Get user's detection data
    user_detections = Detection.objects.filter(user=request.user, is_checked_out=True).order_by('-created_at')
    
    # Write header
    writer.writerow([
        'Date', 'Detection ID', 'Total Items', 'Total Amount (RM)', 
        'Item Name', 'Quantity', 'Unit Price (RM)', 'Item Total (RM)'
    ])
    
    # Write detection data
    for detection in user_detections:
        for item in detection.items.filter(menu_item__isnull=False):
            writer.writerow([
                detection.created_at.strftime('%Y-%m-%d %H:%M'),
                detection.id,
                detection.items.count(),
                f"{float(detection.total_amount):.2f}",
                item.menu_item.name,
                f"{float(item.quantity):.2f}",
                f"{float(item.menu_item.price):.2f}",
                f"{float(item.calculated_price):.2f}"
            ])
    
    return response

@login_required
def menu_price_view(request):
    # Get all menu items and categories
    menu_items = MenuItem.objects.select_related('category').all()
    categories = FoodCategory.objects.prefetch_related('items').all()
    
    # Initialize forms
    form = MenuItemForm()
    category_form = FoodCategoryForm()
    
    # Handle form submissions
    if request.method == 'POST':
        print(f"POST data: {request.POST}")  # Debug: see what's being submitted
        if 'add_item' in request.POST:
            form = MenuItemForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                messages.success(request, 'Menu item added successfully!')
                return redirect('menu_price')
            else:
                print(f"Menu item form errors: {form.errors}")  # Debug: see form errors
                # Show specific field errors in toast notifications
                for field, errors in form.errors.items():
                    field_name = form.fields[field].label or field.replace('_', ' ').title()
                    for error in errors:
                        messages.error(request, f'{field_name}: {error}')
        elif 'add_category' in request.POST:
            category_form = FoodCategoryForm(request.POST)
            if category_form.is_valid():
                category_form.save()
                messages.success(request, 'Category added successfully!')
                return redirect('menu_price')
            else:
                print(f"Category form errors: {category_form.errors}")  # Debug: see form errors
                # Show specific field errors in toast notifications
                for field, errors in category_form.errors.items():
                    field_name = category_form.fields[field].label or field.replace('_', ' ').title()
                    for error in errors:
                        messages.error(request, f'{field_name}: {error}')
        else:
            print("POST request but no recognized form type found")  # Debug: unknown POST
    
    # Statistics
    stats = {
        'total_items': menu_items.count(),
        'available_items': menu_items.filter(is_available=True).count(),
        'total_categories': categories.count(),
        'avg_price': menu_items.aggregate(avg=models.Avg('price'))['avg'] or 0,
        'min_price': menu_items.aggregate(min=models.Min('price'))['min'] or 0,
        'max_price': menu_items.aggregate(max=models.Max('price'))['max'] or 0,
        'total_value': menu_items.aggregate(sum=Sum('price'))['sum'] or 0,
    }
    
    # Order items by category and name
    menu_items = menu_items.order_by('category__name', 'name')
    
    context = {
        'menu_items': menu_items,
        'categories': categories,
        'form': form,
        'category_form': category_form,
        'stats': stats,
        # Individual stats for template
        'available_items_count': stats['available_items'],
        'avg_price': stats['avg_price'],
        'min_price': stats['min_price'],
        'max_price': stats['max_price'],
        'total_value': stats['total_value'],
    }
    return render(request, 'food_recognition/menu_price_enhanced.html', context)

@login_required
def add_category_view(request):
    if request.method == 'POST':
        form = FoodCategoryForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Category added successfully!')
            return redirect('menu_price')
    else:
        form = FoodCategoryForm()
    
    return render(request, 'food_recognition/add_category.html', {'form': form})

@login_required
def edit_category_view(request, category_id):
    category = get_object_or_404(FoodCategory, id=category_id)
    
    if request.method == 'POST':
        form = FoodCategoryForm(request.POST, instance=category)
        if form.is_valid():
            form.save()
            messages.success(request, 'Category updated successfully!')
            return redirect('menu_price')
        else:
            messages.error(request, 'Please correct the errors in the form.')
            return redirect('menu_price')
    
    # For GET requests, redirect back to menu page
    return redirect('menu_price')

@login_required
def delete_category_view(request, category_id):
    category = get_object_or_404(FoodCategory, id=category_id)
    if request.method == 'POST':
        items_count = category.items.count()
        if items_count > 0:
            messages.error(request, f'Cannot delete category with {items_count} items. Please reassign or delete the items first.')
        else:
            category.delete()
            messages.success(request, 'Category deleted successfully!')
    return redirect('menu_price')

@login_required
def edit_menu_item(request, item_id):
    item = get_object_or_404(MenuItem, id=item_id)
    
    if request.method == 'POST':
        form = MenuItemForm(request.POST, request.FILES, instance=item)
        if form.is_valid():
            form.save()
            messages.success(request, 'Menu item updated successfully!')
            return redirect('menu_price')
    else:
        form = MenuItemForm(instance=item)
    
    return render(request, 'food_recognition/edit_menu_item.html', {'form': form, 'item': item})

@login_required
def delete_menu_item(request, item_id):
    item = get_object_or_404(MenuItem, id=item_id)
    if request.method == 'POST':
        item_name = item.name
        item.delete()
        messages.success(request, f'Menu item "{item_name}" deleted successfully!')
    return redirect('menu_price')

@login_required
def duplicate_menu_item(request, item_id):
    original_item = get_object_or_404(MenuItem, id=item_id)
    
    # Create a duplicate
    duplicate = MenuItem(
        name=f"{original_item.name} (Copy)",
        category=original_item.category,
        price=original_item.price,
        pricing_unit=original_item.pricing_unit,
        description=original_item.description,
        nutritional_info=original_item.nutritional_info,
        preparation_time=original_item.preparation_time,
        is_available=False  # Set as unavailable by default
    )
    duplicate.save()
    
    messages.success(request, f'Menu item duplicated successfully! "{duplicate.name}" has been created.')
    return redirect('menu_price')

@login_required
def bulk_availability_update(request):
    if request.method == 'POST':
        available = request.POST.get('available') == 'true'
        MenuItem.objects.all().update(is_available=available)
        status = 'available' if available else 'unavailable'
        messages.success(request, f'All menu items marked as {status}!')
    return redirect('menu_price')

@login_required
def export_menu_view(request):
    import csv
    from django.http import HttpResponse
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="menu_export.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Name', 'Category', 'Price', 'Pricing Unit', 'Description', 'Available', 'Created Date', 'Updated Date'])
    
    menu_items = MenuItem.objects.select_related('category').all()
    for item in menu_items:
        writer.writerow([
            item.name,
            item.category.name if item.category else 'No Category',
            str(item.price),
            item.get_pricing_unit_display(),
            item.description,
            'Yes' if item.is_available else 'No',
            item.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            item.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    return response

def logout_view(request):
    """Custom logout view that redirects to login page"""
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('login')
