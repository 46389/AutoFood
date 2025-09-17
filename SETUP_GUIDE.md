# AutoFood - Enhanced Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- Your YOLOv11-seg trained model file (`best.pt`)

### Installation Steps

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your model file:**
   - Put your `best.pt` model file in the project root directory

4. **Run database migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create superuser (admin account):**
   ```bash
   python manage.py createsuperuser
   ```

6. **Create some sample categories (optional):**
   ```bash
   python manage.py shell
   ```
   Then run:
   ```python
   from food_recognition.models import FoodCategory
   
   categories = [
       {'name': 'Main Course', 'color': '#e74c3c', 'icon': 'fas fa-hamburger'},
       {'name': 'Rice & Grains', 'color': '#f39c12', 'icon': 'fas fa-seedling'},
       {'name': 'Meat & Protein', 'color': '#c0392b', 'icon': 'fas fa-drumstick-bite'},
       {'name': 'Vegetables', 'color': '#27ae60', 'icon': 'fas fa-carrot'},
       {'name': 'Seafood', 'color': '#3498db', 'icon': 'fas fa-fish'},
       {'name': 'Beverages', 'color': '#9b59b6', 'icon': 'fas fa-glass-water'},
       {'name': 'Desserts', 'color': '#e91e63', 'icon': 'fas fa-ice-cream'},
   ]
   
   for cat_data in categories:
       FoodCategory.objects.get_or_create(
           name=cat_data['name'],
           defaults={
               'color': cat_data['color'],
               'icon': cat_data['icon'],
               'description': f'{cat_data["name"]} category for menu organization'
           }
       )
   
   exit()
   ```

7. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

8. **Access the application:**
   - Main application: http://127.0.0.1:8000/
   - Admin interface: http://127.0.0.1:8000/admin/

## 🎨 Enhanced Features

### Interactive UI Elements
- **Glassmorphism Design**: Modern glass effect with backdrop blur
- **Smooth Animations**: CSS transitions and hover effects
- **Interactive Cards**: 3D tilt effects on desktop
- **Live Clock**: Real-time clock in navigation bar
- **Progress Animations**: Animated statistics and progress bars
- **Responsive Design**: Optimized for all device sizes

### Advanced Menu Management
- **Category System**: Organize items by food categories with custom colors and icons
- **Pricing Units**: Support for per item, per gram, per portion, and per serving pricing
- **Search & Filter**: Real-time search with category and availability filters
- **Image Support**: Upload and display menu item images
- **Nutritional Info**: Track calories, protein, and other nutritional data
- **Preparation Time**: Estimate cooking/preparation time for each item
- **Bulk Operations**: Mass update availability status
- **Export/Import**: CSV export functionality for menu data
- **Duplicate Items**: Quick duplication of similar menu items

### Enhanced Detection Features
- **Advanced Camera Controls**: Start/stop camera with preview
- **Real-time Confidence**: Dynamic confidence threshold adjustment
- **Interactive Results**: Edit, add, or remove detected items
- **Smart Pricing**: Automatic price calculation based on detected items
- **Detection History**: Complete audit trail of all detections
- **Export Capabilities**: Generate reports and export data

## 📱 User Interface Highlights

### Navigation
- Fixed top navigation with glass effect
- Active page indicators
- User dropdown with profile options
- Real-time clock display

### Home Page
- Toggle between camera and file upload
- Live camera preview with capture controls
- Drag-and-drop file upload
- Interactive confidence slider
- Quick statistics display

### Menu Management
- Grid view with category tabs
- Advanced search and filtering
- Modal forms for adding/editing
- Bulk operations panel
- Price analytics dashboard
- Category color coding

### Detection Results
- Side-by-side image comparison
- Editable detection items
- Real-time total calculation
- Confidence adjustment tools
- Quick checkout process

## 🔧 Configuration Options

### Model Configuration
Update the model path in `food_recognition/views.py`:
```python
# Line 121
model_path = 'best.pt'  # Your YOLOv11-seg model file
```

### UI Customization
Modify colors and styling in `base.html`:
```css
:root {
    --primary-color: #2563eb;    /* Blue */
    --secondary-color: #10b981;  /* Green */
    --accent-color: #f59e0b;     /* Yellow */
    --danger-color: #ef4444;     /* Red */
}
```

### Performance Settings
For production deployment, update `settings.py`:
```python
DEBUG = False
ALLOWED_HOSTS = ['yourdomain.com']
STATIC_ROOT = BASE_DIR / 'staticfiles'
```

## 📊 Database Schema

### Models Overview
- **FoodCategory**: Categories with colors and icons
- **MenuItem**: Enhanced menu items with detailed information
- **Detection**: Image analysis sessions
- **DetectedItem**: Individual detected food items

### Key Features
- Foreign key relationships for data integrity
- Cascade deletion for clean data management
- Indexed fields for fast queries
- Rich metadata support

## 🎯 Usage Tips

### Getting Started
1. Create food categories first (Admin → Food Categories)
2. Add menu items to categories (Menu & Pricing page)
3. Test detection with sample images
4. Adjust confidence thresholds for optimal results

### Best Practices
- Use high-quality menu item images (200x200px recommended)
- Set realistic preparation times
- Keep nutritional information updated
- Regularly export menu data as backup
- Monitor detection accuracy and adjust confidence levels

### Troubleshooting
- If camera doesn't work, check browser permissions
- For slow performance, reduce image resolution
- If models don't load, verify `best.pt` file location
- Check Django logs for detailed error information

## 🚀 Production Deployment

### Required Changes
1. Set `DEBUG = False` in settings.py
2. Configure proper database (PostgreSQL recommended)
3. Set up static file serving (WhiteNoise included)
4. Configure environment variables for secrets
5. Set up proper media file handling

### Security Considerations
- Change Django SECRET_KEY for production
- Set up HTTPS with SSL certificates
- Configure proper user permissions
- Regular security updates for dependencies
- Backup database and media files regularly

## 📈 Performance Optimization

### Frontend Optimizations
- CSS and JavaScript minification
- Image compression and optimization
- Lazy loading for images
- Progressive Web App (PWA) features

### Backend Optimizations
- Database query optimization
- Caching with Redis (optional)
- Background task processing
- API rate limiting

## 🤝 Support

For issues and feature requests:
1. Check the Django error logs
2. Verify all dependencies are installed
3. Ensure model file is accessible
4. Review browser console for JavaScript errors

The application includes comprehensive error handling and user-friendly feedback messages to guide users through any issues.

---

**Enjoy your enhanced AutoFood experience! 🍽️✨**