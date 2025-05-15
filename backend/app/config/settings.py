DATABSES = {
    'default': {
        'ENGINE':'django.db.backends.mysql',
        'NAME': 'django_user',
        'USER': 'root',
        'PASSWORD': 'raouf124',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        }
        
    }
    
}
