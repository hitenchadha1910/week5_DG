web: gunicorn flask_wine_app:app --log-file - --log-level debug
heroku ps:scale web=1
python manage.py migrate
