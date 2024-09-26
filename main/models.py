from django.db import models

# Create your models here.


from django.contrib.auth.models import AbstractBaseUser, BaseUserManager,AbstractUser

#from main import admin


class UserManager(BaseUserManager):
    def create_user(self, name, password=None, **extra_fields):
        if not name:
            raise ValueError('The Name field must be set')

        # phone_number 값을 password로 사용
        extra_fields.setdefault('phone_number', password)

        user = self.model(name=name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, name, password=None, **extra_fields):
        extra_fields.setdefault('is_admin', True)
        return self.create_user(name, password, **extra_fields)

    def __str__(self):
        return f"SurveyResponse for {self.user.name}"

    def survey_form_submit(self, q1, q2, q3, q4, q5, q6, q7):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7
        self.save(using=self._db)

class User(AbstractBaseUser):
    name = models.CharField(max_length=255, unique=False)
    # phone_number = models.CharField(max_length=20, unique=False)  # phone_number 필드로 사용
    last_login = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    q1 = models.CharField(max_length=20, default='',null=True)
    q2 = models.CharField(max_length=20, default='',null=True)
    q3 = models.CharField(max_length=20, default='',null=True)
    q4 = models.CharField(max_length=20, default='',null=True)
    q5 = models.CharField(max_length=10, default='',null=True)
    q6 = models.CharField(max_length=10, default='',null=True)
    q7 = models.CharField(max_length=10, default='',null=True)
    objects = UserManager()

    def survey_form_submit(self, q1, q2, q3, q4, q5, q6, q7):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.q6 = q6
        self.q7 = q7
        self.save()

    USERNAME_FIELD = 'name'
    REQUIRED_FIELDS = []





# class SurveyResponse(models.Model):
#     # user = models.ForeignKey(User, on_delete=models.CASCADE)
#     # q1 = models.CharField(max_length=20)
#     # q2 = models.CharField(max_length=20)
#     # q3 = models.CharField(max_length=20)
#     # q4 = models.CharField(max_length=20)
#     # q5 = models.CharField(max_length=10)
#     # q6 = models.CharField(max_length=10)
#     # q7 = models.CharField(max_length=10)
#     # objects = UserManager()




