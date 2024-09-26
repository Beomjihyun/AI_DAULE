from django.urls import path
from .import views
from .views import survey_submit


urlpatterns = [
    path('question1/', views.questions1, name='question1'),
    path('question2/', views.questions2, name='question2'),
    path('question3/', views.questions3, name='question3'),
    path('question4/', views.questions4, name='question4'),
    path('question5/', views.questions5, name='question5'),
    path('question6/', views.questions6, name='question6'),
    path('question7/', views.questions7, name='question7'),
    path('login/index/', views.index, name='index'),
    path('login/', views.login_register, name='login'),
    path('', views.main),
    path('result/', views.survey_submit, name='survey_submit'),
    #path('result/', views.result),
    #path('result/survey/', views.survey_form, name='survey_form'),

]
