# import all necessary 3rd python packages 
from copy import copy
from flask import Flask, render_template, Response, request, url_for, redirect, flash
import modules.mysql_connection

from faceMeshProjectForFACE_OVAL import faceMeshDetection, FaceMeshDetector
# from app3 import faceMeshDetection
# from takePicture0310 import streamlive,facePicture
from takePictureCountdown import streamlive
# from takePicture0314_IP import streamlive

import cv2
import sqlQuery

from flask_table import Table, Col, LinkCol

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import configparser

# from faceMeshProjectForFlask import faceMeshDetection_test
#
# Flask 類別 初始化時 傳入的 __name__ 參數，代表當前模組的名稱。
# 為固定用法，以便讓 Flask 知道在哪裡尋找資源。
# (例如: 模板和靜態文件)

# app = Flask(__name__, '/')
import flask
# from flask_cors import CORS

app = flask.Flask(__name__, '/')
# CORS(app)

### login
# config 初始化
config = configparser.ConfigParser()
config.read('config.ini')

# app.secret_key = config.get('flask', 'secret_key')
app.secret_key = 'dd06be55a06c03312b2ab109b5f8f6ab'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
login_manager.login_view = 'login'
# login_manager.login_message = '請證明你並非來自黑暗草泥馬界'
login_manager.login_message = 'YAYAYA'

class User(UserMixin):
    pass

@login_manager.user_loader
def user_loader(使用者):
    # if 使用者 not in users:
    # if 使用者 is None or sqlQueryMember(使用者, ''):
        # return

    user = User()
    user.id = 使用者
    return user

@login_manager.request_loader
def request_loader(request):
    使用者 = request.form.get('user_id')
    # if 使用者 not in users:
    if 使用者 is None or sqlQueryMember(使用者, ''):
        return

    user = User()
    user.id = 使用者

    # DO NOT ever store passwords in plaintext and always compare password
    # hashes using constant-time comparison!
    # user.is_authenticated = request.form['password'] == users[使用者]['password']
    user.is_authenticated = sqlQueryMember(使用者, request.form['password'])

    return user

# users = {'Me': {'password': 'myself'}, '123': {'password': '456'}}

# sql 查詢 member, 回傳 True or False
def sqlQueryMember(account_number, password):
    if account_number is None:
        return False
    memberExist = sqlQuery.sqlQueryMember(account_number, password)
    print(memberExist)
    # return True
    return memberExist

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template("login.html")
    
    使用者 = request.form['user_id']
    # if (使用者 in users) and (request.form['password'] == users[使用者]['password']):
    if sqlQueryMember(使用者, request.form['password']):
        user = User()
        user.id = 使用者
        login_user(user)
        # flash(f'{使用者}！歡迎加入草泥馬訓練家的行列！')
        flash(f'{使用者}！YAYAYA！')
        return redirect(url_for('index'))

    flash('登入失敗了...')
    return render_template('login.html')

@app.route('/logout')
def logout():
    使用者 = current_user.get_id()
    logout_user()
    flash(f'{使用者}！歡迎下次再來！')
    return render_template('login.html')
### login 結束

# 裝飾器是告訴 Flask，哪個 URL 應該觸發我們的函式。
# 斜線代表的就是網站的根目錄，可以疊加。
# 例如: 新增一個 /hello 的位址
# @app.route("/")
# @app.route("/hello")
# def hello():
#     return "Hello, World!"
# 網站訪問首頁與/hello，呈現的同樣是 hello 函式，回傳的 Hello World 文字。
#render_template是flask的內建函數，字面上的意思大概是渲染模板，因此它需要一個『模板』來提供渲染的格式。
# 當return render_template('html文件')的時候，flask會先到專案資料夾『templates』去尋找相對應的html文件，
# 因此，你需要先做的就是在專案底下建置一個資料夾，並且命名為『templates』
@app.route('/')
# @login_required
def index():
    return render_template('Home.html')
    # return render_template('Index.html')
    # return render_template('Test.html')

#功能體驗
@app.route('/Experience')
def Experience():
    return render_template('Experience.html')

#數據分析
@app.route('/DataAnalysis')
@login_required
def DataAnalysis():
    return render_template('DataAnalysis.html')  

#美學標準
@app.route('/Aesthetics')
def Aesthetics():
    return render_template('Aesthetics.html')   

#關於我們
@app.route('/About')
def About():
    return render_template('About.html')

#登入
@app.route('/Signin')
def Signin():
    return render_template('Signin.html')  

#註冊
@app.route('/Signup')
def Signup():
    return render_template('Signup.html')

# feed a video stream as a source
# 前端的 video_feed 加上 style 參數, 傳到 drawFortuneTelling, 表示要畫哪一種圖; 加上 videoMode 參數, 傳到 videoMode, 表示"影片"或"照片"
@app.route('/video_feed/<string:style>/<string:videoMode>')
def video_feed(style, videoMode):
    # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
    # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
    # The header is old and widely supported. It works in legacy browsers.
    # Each yield expression is directly sent thru Response to the browser.

    # 是否為影片
    # videoMode = False
    # videoMode = True
    # print(videoMode)

    # 路徑
    # filePath = "./videos/1-720p.mp4"
    # filePath = "app/static/images/Thelatestphotos.jpg"
    
    if videoMode == "影片":
        filePath = "./videos/1-720p.mp4"
        videoMode = True
    elif videoMode == "照片":
        # filePath = "app/static/images/Thelatestphotos.jpg"
        # 大餅臉
        filePath = "app/static/images/Thelatestphotos.jpg"
        # AngelaBaby
        # filePath = "app/static/Thelatestphotos7.jpg"
        # 側臉
        # filePath = "app/static/Thelatestphotos8.jpg"
        # "完美大帥哥
        # filePath = "app/static/handsome.jpg"
        # 小夫(側臉線條失敗)
        # filePath = "app/static/Thelatestphotos11.jpg"

        videoMode = False
    elif videoMode == "照相":
        return Response(streamlive(style),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    drawFaceLms = True 
    drawID = False 
    drawFortuneTelling = style
    return Response(faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    # mimetype 媒體類別 multipart/x-mixed-replace 資料傳輸格式


@app.route('/getTxt', methods=["POST"])
def getTxt():
    #print(flask.request.json["param"])
    # 是否為影片
    videoMode = False
    # videoMode = True
    # 路徑
    filePath = "app/static/images/Thelatestphotos.jpg"
    img = cv2.imread(filePath)
    # filePath = "./videos/1-720p.mp4"

    drawFaceLms = True 
    drawID = False 
    drawFortuneTelling = flask.request.json["param"]

    returnTxt=True
    # returnComment: 回傳評論
    returnComment=True
    # returnComment=False

    # txt = "YAYAYA"
    # txt = faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling, returnTxt)
    txt, comment, hair_makeup_comment = FaceMeshDetector(maxFaces=10).findFaceMesh(img.copy(), drawFaceLms=True, drawID=False, drawFortuneTelling=drawFortuneTelling, returnTxt=returnTxt, returnComment=returnComment)
    # print(findFaceMesh(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling, returnTxt))
    # print(txt)
    # print(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling, returnTxt)

    data={ "回傳文字":txt, "回傳評論": comment, "回傳髮型妝容建議": hair_makeup_comment}
    return flask.jsonify(data)

# 老師示範用 ajax 前後端傳送資料
@app.route('/getFaceData', methods=["POST"])
def getFaceData():
    #print(flask.request.json["param"])
    data={ "臉部周長":flask.request.json["param"]**2}    
    return flask.jsonify(data)

# # stream live
# @app.route('/stream_live/<string:style>', methods=["GET"])
# def stream_live(style):          
#     # photograph = style
#     return Response(streamlive(style),
#                 mimetype='multipart/x-mixed-replace; boundary=frame')
#     # mimetype 媒體類別 multipart/x-mixed-replace 資料傳輸格式

# 連接AZURE的mysql
# @app.route('/try-mysql')
# def try_mysql():
#     (cursor, cnx) = modules.mysql_connection.get_cursor()
#     sql = ("SELECT * FROM new_table")
#     cursor.execute(sql)
#     return render_template('data_table.html', t_data=cursor.fetchall())

@app.route('/try-mysql')
def try_mysql():
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = ("SELECT * FROM new_table")
    cursor.execute(sql)
    # return render_template('data_table.html', t_data=cursor.fetchall())
    rows = cursor.fetchall()
    table = Results(rows)
    table.border = True

    param = 1
    sql = (f"SELECT * FROM new_table where face_id = {param}")
    cursor.execute(sql)
    # rows = cursor.fetchall()
    # table = Results(rows)
    # table.border = True
    t_data=cursor.fetchall()
    print(t_data)
    return render_template('data_table.html', table=table, t_data=t_data)

class Results(Table):
    # user_id = Col('Id', show=False)
    face_id = Col('face_id')
    face_name = Col('face_name')
    # edit = LinkCol('Edit', 'edit_view', url_kwargs=dict(id='face_id'))
    # delete = LinkCol('Delete', 'delete_user', url_kwargs=dict(id='face_id'))

'''
@app.route('/mysql')
def mysqll():
    (cursor, cnx) = modules.mysql_connection.get_cursor()
    sql = ("SELECT * FROM address_book LIMIT 5")
    cursor.execute(sql)
    data = cursor.fetchall()
    # return str(data)
    # return jsonify(data)    
    return render_template('members.html', data = data)
'''
#
# end of program