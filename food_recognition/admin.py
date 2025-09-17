from django.contrib import admin
from .models import FoodCategory, MenuItem, Detection, DetectedItem

@admin.register(FoodCategory)
class FoodCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'color', 'icon', 'created_at', 'item_count')
    list_filter = ('created_at',)
    search_fields = ('name', 'description')
    ordering = ('name',)
    
    def item_count(self, obj):
        return obj.items.count()
    item_count.short_description = 'Number of Items'

@admin.register(MenuItem)
class MenuItemAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'price', 'pricing_unit', 'is_available', 'created_at')
    list_filter = ('category', 'pricing_unit', 'is_available', 'created_at')
    search_fields = ('name', 'description')
    list_editable = ('price', 'is_available')
    ordering = ('category__name', 'name')
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'category', 'description', 'image')
        }),
        ('Pricing', {
            'fields': ('price', 'pricing_unit')
        }),
        ('Details', {
            'fields': ('is_available',),
            'classes': ('collapse',)
        })
    )

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'confidence_threshold', 'created_at', 'is_checked_out', 'total_amount', 'item_count')
    list_filter = ('is_checked_out', 'created_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at', 'checkout_date')
    ordering = ('-created_at',)
    
    def item_count(self, obj):
        return obj.items.count()
    item_count.short_description = 'Detected Items'

@admin.register(DetectedItem)
class DetectedItemAdmin(admin.ModelAdmin):
    list_display = ('detection', 'class_label', 'confidence', 'quantity', 'menu_item', 'is_manually_added')
    list_filter = ('is_manually_added', 'detection__created_at', 'menu_item__category')
    search_fields = ('class_label', 'menu_item__name')
    ordering = ('-detection__created_at',)
