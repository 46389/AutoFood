from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Authentication URLs
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('register/', views.register_view, name='register'),
    
    # Main application URLs
    path('', views.home_view, name='home'),
    path('process-image/', views.process_image, name='process_image'),
    path('results/<int:detection_id>/', views.detection_results, name='detection_results'),
    
    # Detection editing URLs
    path('edit-item/<int:item_id>/', views.edit_detection, name='edit_detection'),
    path('delete-item/<int:item_id>/', views.delete_detection, name='delete_detection'),
    path('add-item/<int:detection_id>/', views.add_manual_item, name='add_manual_item'),
    path('update-item-quantity/', views.update_item_quantity, name='update_item_quantity'),
    
    # Checkout and reset URLs
    path('checkout/<int:detection_id>/', views.checkout, name='checkout'),
    path('reset/<int:detection_id>/', views.reset_detection, name='reset_detection'),
    
    # Dashboard and menu URLs
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('export-dashboard/', views.export_dashboard_data, name='export_dashboard'),
    path('menu-price/', views.menu_price_view, name='menu_price'),
    
    # Menu item management
    path('edit-menu/<int:item_id>/', views.edit_menu_item, name='edit_menu_item'),
    path('delete-menu/<int:item_id>/', views.delete_menu_item, name='delete_menu_item'),
    path('duplicate-menu/<int:item_id>/', views.duplicate_menu_item, name='duplicate_menu_item'),
    
    # Category management
    path('add-category/', views.add_category_view, name='add_category'),
    path('edit-category/<int:category_id>/', views.edit_category_view, name='edit_category'),
    path('delete-category/<int:category_id>/', views.delete_category_view, name='delete_category'),
    
    # Bulk operations
    path('bulk-availability/', views.bulk_availability_update, name='bulk_availability'),
    path('export-menu/', views.export_menu_view, name='export_menu'),
]