# AI黃金面容分析應用


 * 使用Google Mediapipe Face Mesh </br>
 
 * 使用者可直接透過前端網頁進行拍照 </br>
 
 * 利用Mediapipe畫座標代號再加以畫線 </br>
 
 * 依據臉部三庭五眼等比例角度的美學面相 </br>
 
 * 計算出相對應長度或比例數據 </br>
 
 * 依據比例數據從MySQL資料庫評價 </br>
 
 * 依照數據去推薦適合使用者的髮型及妝容 </br>
 
![image](https://i.imgur.com/IyPKp9n.jpg)


 # 使用套件
 ` $ pip install -r requirements.txt `
 * Flask==2.0.3
 * Flask-Babel==2.0.0
 * Flask-Login==0.5.0
 * Flask-Table==0.5.0
 * Jinja2==3.0.3
 * opencv-contrib-python==4.5.5.62
 * opencv-python==4.5.5.62
 * mediapipe==0.8.9.1
 * numpy==1.21.5
 * python==3.8
 * numpy==1.21.5
 * mysql-connector-python==8.0.28
 * mysqlclient==2.1.0
  
 # install
 
 * git clone this
 * 建一個資料夾videos
 * 放進人臉影片並命名1-720p.mp4
 * 安裝requirements.txt的相依性套件
 * 匯入 aien16.sql 到MySQL
 * 匯出 requirements.txt的指令 `$ pip freeze > requirements.txt`
 * ` $ start.bat `
 
 # mediapipe faceid
 ![image](https://i.imgur.com/5PvuFlq.png)




