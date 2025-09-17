from django.core.management.base import BaseCommand
from food_recognition.models import FoodCategory, MenuItem
from decimal import Decimal
import random

class Command(BaseCommand):
    help = 'Create menu items from trained model classes'
    
    def handle(self, *args, **options):
        # Food items from your trained model
        food_items = [
            'amaranthus', 'ayam kunyit', 'ayam masak merah', 'bak choy', 'bean sprout',
            'beancurd skin', 'beef rendang', 'bitter gourd', 'boiled egg', 'braised egg',
            'braised meat', 'brinjal dish', 'broccoli', 'butter chicken', 'cabbage',
            'chayote', 'chicken with mushroom', 'chilli', 'chinese cabbage', 'cucumber',
            'curry meat', 'dendang batokok', 'fish', 'fish curry', 'fried chicken',
            'fried egg', 'fried mushroom', 'fried pork meat', 'kailan', 'kam heong meat',
            'kangkong', 'ladyfinger', 'lettuce', 'long bean', 'luncheon meat',
            'meat with black fungus', 'noodles', 'omelette', 'pork with salted vegetable',
            'potato', 'prawn', 'rice', 'roasted chicken', 'sambal egg', 'sausage',
            'scrambled egg with tomato', 'sesame chicken', 'sotong', 'soy chicken',
            'spinash', 'steamed chicken', 'steamed egg', 'steamed minced meat',
            'stir-fried meat', 'sweet and sour meat', 'sweet potato leaves', 'tofu', 'winged bean'
        ]
        
        # Category mapping
        category_mapping = {
            'Rice & Noodles': ['rice', 'noodles'],
            'Meat': [
                'ayam kunyit', 'ayam masak merah', 'beef rendang', 'butter chicken',
                'chicken with mushroom', 'fried chicken', 'roasted chicken', 
                'sesame chicken', 'soy chicken', 'steamed chicken', 'curry meat', 
                'dendang batokok', 'braised meat', 'fried pork meat', 'kam heong meat', 
                'meat with black fungus', 'pork with salted vegetable', 'stir-fried meat', 
                'sweet and sour meat', 'steamed minced meat', 'luncheon meat', 'sausage'
            ],
            'Seafood': ['fish', 'fish curry', 'prawn', 'sotong'],
            'Egg': [
                'boiled egg', 'braised egg', 'fried egg', 'omelette', 'sambal egg', 
                'scrambled egg with tomato', 'steamed egg'
            ],
            'Tofu': ['tofu', 'beancurd skin'],
            'Vegetables': [
                'amaranthus', 'bak choy', 'bean sprout', 'bitter gourd', 'broccoli',
                'cabbage', 'chayote', 'chilli', 'chinese cabbage', 'cucumber',
                'kailan', 'kangkung', 'ladyfinger', 'lettuce', 'long bean',
                'potato', 'spinash', 'sweet potato leaves', 'winged bean', 'brinjal dish',
                'fried mushroom'
            ]
        }
        
        # Get categories
        categories = {cat.name: cat for cat in FoodCategory.objects.all()}
        
        created_count = 0
        for food_item in food_items:
            # Find appropriate category
            category = None
            for cat_name, items in category_mapping.items():
                if food_item in items and cat_name in categories:
                    category = categories[cat_name]
                    break
            
            # Generate realistic pricing
            if any(meat in food_item for meat in ['chicken', 'beef', 'pork', 'fish', 'prawn', 'sotong']):
                price = Decimal(str(round(random.uniform(8.0, 25.0), 2)))
                pricing_unit = 'portion'
            elif 'egg' in food_item:
                price = Decimal(str(round(random.uniform(2.0, 6.0), 2)))
                pricing_unit = 'item'
            elif food_item == 'rice':
                price = Decimal(str(round(random.uniform(1.5, 3.0), 2)))
                pricing_unit = 'serving'
            elif food_item == 'noodles':
                price = Decimal(str(round(random.uniform(6.0, 12.0), 2)))
                pricing_unit = 'portion'
            else:  # vegetables and others
                price = Decimal(str(round(random.uniform(3.0, 8.0), 2)))
                pricing_unit = 'portion'
            
            # Create formatted name
            formatted_name = food_item.replace('_', ' ').title()
            
            menu_item, created = MenuItem.objects.get_or_create(
                name=formatted_name,
                defaults={
                    'category': category,
                    'price': price,
                    'pricing_unit': pricing_unit,
                    'description': f'Delicious {formatted_name.lower()} prepared with authentic flavors',
                    'is_available': True,
                    'preparation_time': random.randint(10, 30)
                }
            )
            
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created menu item: {formatted_name} - ${price}/{pricing_unit}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Menu item already exists: {formatted_name}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {created_count} new menu items.')
        )