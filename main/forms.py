from django import forms
from .models import User

class LoginForm(forms.Form):
    login_name = forms.CharField(label='이름')
    login_phone_number = forms.CharField(label='전화번호', widget=forms.PasswordInput)

    def clean_phone_number(self):
        phone_number = self.cleaned_data['phone_number']

        # 중복된 번호가 있는지 확인하는 로직을 추가합니다.
        if User.objects.filter(phone_number=phone_number).exists():
            raise forms.ValidationError("이미 등록된 전화번호입니다.")

        return phone_number

