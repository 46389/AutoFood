from django.core.management.base import BaseCommand
from food_recognition.models import FoodCategory

class Command(BaseCommand):
    help = 'Create default food categories with green theme colors'
    
    def handle(self, *args, **options):
        categories = [
            {'name': 'Meat', 'color': '#8AA624', 'icon': 'fas fa-drumstick-bite', 'description': 'Pork, chicken, beef, and other meat dishes'},
            {'name': 'Seafood', 'color': '#DBE4C9', 'icon': 'fas fa-fish', 'description': 'Fish, prawn, sotong, and seafood dishes'},
            {'name': 'Egg', 'color': '#FEA405', 'icon': 'fas fa-egg', 'description': 'Egg preparations and dishes'},
            {'name': 'Tofu', 'color': '#FFFFF0', 'icon': 'fas fa-cube', 'description': 'Tofu and soy-based protein dishes'},
            {'name': 'Vegetables', 'color': '#22c55e', 'icon': 'fas fa-carrot', 'description': 'Fresh vegetables and salads'},
            {'name': 'Rice & Noodles', 'color': '#34d399', 'icon': 'fas fa-seedling', 'description': 'Rice, noodles, and grain-based dishes'},
        ]
        
        created_count = 0
        for cat_data in categories:
            category, created = FoodCategory.objects.get_or_create(
                name=cat_data['name'],
                defaults={
                    'color': cat_data['color'],
                    'icon': cat_data['icon'],
                    'description': cat_data['description']
                }
            )
            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Created category: {category.name}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Category already exists: {category.name}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {created_count} new categories.')
        )