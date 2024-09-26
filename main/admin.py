from django.contrib import admin
from .models import User

# Register your models here.

# admin.site.register(User)

class UserAdmin(admin.ModelAdmin):
    list_display = ['name', 'password', 'q1','q2','q3','q4','q5','q6','q7']  # Specify the fields to display in the list view

admin.site.register(User, UserAdmin)