from django.shortcuts import render, redirect
from .models import Profile, Employee
from django.contrib import messages
from django.contrib.auth.models import User, auth
from .forms import EmployeeForm
from django.contrib.sessions.models import Session
import datetime
#create your views here


def UI(requests):
    #em = Employee.objects.values('id','name')  just a trial
    #names = list(em)
    # date = now()

    return render(requests, 'UI/abc.html')

def sec(requests):

    prof = Profile.objects.all() #for fetching data from database

    return render(requests, 'UI/second.html', {'prof': prof})

def login(requests):

    if requests.method== 'POST':
        username = requests.POST['username']
        password = requests.POST['password']

        user = auth.authenticate(username=username,password=password)
        
        if user is not None:
            #requests.session['user_username'] = True
            auth.login(requests, user)
            return redirect('/register')
        else:
            messages.info(requests,'oops!! invalid credentials')
            return redirect('/login')    
    else:
        return render(requests,'UI/third.html')  


def fou(requests):
    return render(requests, 'UI/fourth.html')

def reg(requests):
    if requests.user.is_authenticated:
        if requests.method == 'POST':
            emp_form = EmployeeForm(requests.POST)
            if emp_form.is_valid():
                name = requests.POST['name']
                phone = requests.POST['phone']
                emp = Employee(name=name, phone=phone)
                emp.save()
        else:        
            emp_form = EmployeeForm()
        
        context = {
            'form' : emp_form
        }    
        return render(requests, 'UI/fifth.html', context)
    else:
        return redirect('/login') 
    

     

  
def logout(requests):
    auth.logout(requests)
    # try:
    #     del requests.session['user_username']
    # except KeyError:
    #     pass
    return redirect('/')


