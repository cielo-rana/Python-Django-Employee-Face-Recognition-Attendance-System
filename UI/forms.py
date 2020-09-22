from django import forms
from .models import Employee

class EmployeeForm(forms.Form):
    class Meta:
        model = Employee
        fields = [
            'name',
            'phone'
        ]

    

