from flask import Flask, request, abort, render_template, url_for, flash, redirect

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

import configparser

app = Flask(__name__)

# config 初始化
config = configparser.ConfigParser()
config.read('config.ini')
app.secret_key = 'dd06be55a06c03312b2ab109b5f8f6ab'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
login_manager.login_view = 'login'
login_manager.login_message = '請證明你並非來自黑暗草泥馬界'

class User(UserMixin):
    pass

@login_manager.user_loader
def user_loader(使用者):
    if 使用者 not in users:
        return

    user = User()
    user.id = 使用者
    return user

@login_manager.request_loader
def request_loader(request):
    使用者 = request.form.get('user_id')
    if 使用者 not in users:
        return

    user = User()
    user.id = 使用者

    # DO NOT ever store passwords in plaintext and always compare password
    # hashes using constant-time comparison!
    user.is_authenticated = request.form['password'] == users[使用者]['password']

    return user

users = {'Me': {'password': 'myself'}}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    
    使用者 = request.form['user_id']
    if (使用者 in users) and (request.form['password'] == users[使用者]['password']):
        user = User()
        user.id = 使用者
        login_user(user)
        flash(f'{使用者}！歡迎加入草泥馬訓練家的行列！')
        return redirect(url_for('from_start'))

    flash('登入失敗了...')
    return render_template('login.html')

@app.route('/logout')
def logout():
    使用者 = current_user.get_id()
    logout_user()
    flash(f'{使用者}！歡迎下次再來！')
    return render_template('login.html')
