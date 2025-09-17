from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import MenuItem, DetectedItem, FoodCategory

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*'
        })
    )
    confidence_threshold = forms.FloatField(
        initial=0.5,
        min_value=0.1,
        max_value=1.0,
        step_size=0.1,
        widget=forms.NumberInput(attrs={
            'type': 'range',
            'class': 'form-range',
            'min': '0.1',
            'max': '1.0',
            'step': '0.1'
        })
    )

class FoodCategoryForm(forms.ModelForm):
    class Meta:
        model = FoodCategory
        fields = ['name', 'description', 'color', 'icon']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 2}),
            'color': forms.TextInput(attrs={'class': 'form-control', 'type': 'color'}),
            'icon': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., fas fa-pizza-slice'}),
        }

class MenuItemForm(forms.ModelForm):
    class Meta:
        model = MenuItem
        fields = ['name', 'category', 'price', 'pricing_unit', 'description', 'image', 'is_available']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'category': forms.Select(attrs={'class': 'form-select'}),
            'price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0'}),
            'pricing_unit': forms.Select(attrs={'class': 'form-select'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'}),
            'is_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

class MenuSearchForm(forms.Form):
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search menu items...',
            'id': 'search-input'
        })
    )
    category = forms.ModelChoiceField(
        queryset=FoodCategory.objects.all(),
        required=False,
        empty_label="All Categories",
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'category-filter'})
    )
    available_only = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'available-filter'})
    )

class DetectedItemForm(forms.ModelForm):
    class Meta:
        model = DetectedItem
        fields = ['class_label', 'quantity', 'menu_item']
        widgets = {
            'class_label': forms.TextInput(attrs={'class': 'form-control'}),
            'quantity': forms.NumberInput(attrs={'class': 'form-control', 'min': '0.1', 'step': '0.1'}),
            'menu_item': forms.Select(attrs={'class': 'form-select'}),
        }

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})