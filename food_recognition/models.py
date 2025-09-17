from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class FoodCategory(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#007bff', help_text='Hex color code for category')
    icon = models.CharField(max_length=50, default='fas fa-utensils', help_text='FontAwesome icon class')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Food Categories"
    
    def __str__(self):
        return self.name

class MenuItem(models.Model):
    PRICING_UNIT_CHOICES = [
        ('item', 'Per Item'),
        ('serving', 'Per Serving'),
    ]
    
    name = models.CharField(max_length=100, unique=True)
    category = models.ForeignKey(FoodCategory, on_delete=models.CASCADE, related_name='items', null=True, blank=True)
    price = models.DecimalField(max_digits=8, decimal_places=2)
    pricing_unit = models.CharField(max_length=20, choices=PRICING_UNIT_CHOICES, default='item')
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='menu_items/', blank=True, null=True)
    is_available = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['category', 'name']
    
    def __str__(self):
        return f"{self.name} - RM{self.price}/{self.get_pricing_unit_display()}"
    
    def get_formatted_price(self):
        return f"RM{self.price}/{self.get_pricing_unit_display().lower()}"

class Detection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_image = models.ImageField(upload_to='uploads/')
    segmented_image = models.ImageField(upload_to='predictions/', blank=True)
    confidence_threshold = models.FloatField(default=0.5)
    created_at = models.DateTimeField(auto_now_add=True)
    is_checked_out = models.BooleanField(default=False)
    checkout_date = models.DateTimeField(null=True, blank=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    def __str__(self):
        return f"Detection {self.id} by {self.user.username}"
    
    def get_total_price(self):
        """Calculate total price of all detected items"""
        return sum(item.get_total_price() for item in self.items.all())
    
    def get_calculated_total_price(self):
        """Get total from calculated_price field"""
        return sum(float(item.calculated_price) for item in self.items.all())

class DetectedItem(models.Model):
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE, related_name='items')
    menu_item = models.ForeignKey(MenuItem, on_delete=models.CASCADE, null=True, blank=True)
    class_label = models.CharField(max_length=100)
    confidence = models.FloatField()
    quantity = models.FloatField(default=1.0)
    bbox_x = models.FloatField()
    bbox_y = models.FloatField()
    bbox_width = models.FloatField()
    bbox_height = models.FloatField()
    mask_data = models.TextField(blank=True)
    estimated_grams = models.FloatField(default=0.0)
    calculated_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    is_manually_added = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.class_label} ({self.confidence:.2f})"
    
    def get_display_quantity(self):
        """Get quantity formatted for display based on menu item pricing unit"""
        if self.menu_item and self.menu_item.pricing_unit == 'serving':
            return f"{self.quantity:.1f} servings ({self.estimated_grams:.0f}g)"
        else:
            return f"{self.quantity:.1f} items ({self.estimated_grams:.0f}g)"
    
    def get_total_price(self):
        """Get total price for this detected item"""
        if self.menu_item:
            return float(self.menu_item.price) * float(self.quantity)
        return 0
