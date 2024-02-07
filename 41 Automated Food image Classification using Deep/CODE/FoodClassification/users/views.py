import os

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def user_predict_food(request):
    if request.method == 'POST':
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        from .utility.Load_Food_Models import start_process
        results = start_process(filename)
        return render(request, "users/which_food_it_is.html", {"path": uploaded_file_url, "results": results})
    else:
        return render(request, "users/which_food_it_is.html", {})


def train_fun(request):
    return render(request, "users/train_func.html", {})


def loss_fun(request):
    return render(request, "users/loss_func.html", {})


def prediction_classes(request):
    label_path = os.path.join(settings.MEDIA_ROOT, 'models', 'labels.txt')
    with open(label_path, 'r') as f:
        food101 = [l.strip().lower() for l in f]
    print(type(food101))
    return render(request, "users/prediction_class.html", {"labels": food101})
