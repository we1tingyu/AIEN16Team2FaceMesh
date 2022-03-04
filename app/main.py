# import all necessary 3rd python packages 
from flask import Flask, render_template, Response
#import modules.mysql_connection
from faceMeshProjectForFlask import faceMeshDetection
#
app = Flask(__name__, '/')
#
@app.route('/')
def index():
    return render_template('faceDetection.html')
#
# feed a video stream as a source
@app.route('/video_feed')
def video_feed():
    # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
    # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
    # The header is old and widely supported. It works in legacy browsers.
    # Each yield expression is directly sent thru Response to the browser.
    videoMode = True
    filePath = "./videos/1-720p.mp4"
    drawFaceLms = True 
    drawID = False 
    drawFortuneTelling = False 
    return Response(faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling),
                mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1():
    # multipart/x-mixed-replace is an HTTP header. Your server can use it to push dynamically updated content to the web browser.
    # It works by telling the browser to keep the connection open and replace the web page or piece of media it is displaying with another when it receives a special token.
    # The header is old and widely supported. It works in legacy browsers.
    # Each yield expression is directly sent thru Response to the browser.
    videoMode = True
    filePath = "./videos/1-720p.mp4"
    drawFaceLms = False 
    drawID = False 
    drawFortuneTelling = True 
    return Response(faceMeshDetection(videoMode, filePath, drawFaceLms, drawID, drawFortuneTelling),
                mimetype='multipart/x-mixed-replace; boundary=frame')

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