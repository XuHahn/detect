from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import timedelta
import os
import predict
import cv2
app = Flask(__name__)

# 输出
@app.route('/')
def hello_world():
    return render_template('upload.html')

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 添加路由
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        # 当前文件所在路径
        basepath = os.path.dirname(__file__)
        # 一定要先创建该文件夹，不然会提示没有该路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        # 保存文件
        f.save(upload_path)
        # 返回上传成功界面
        img = cv2.imread(upload_path)
        path = predict.draw_img(img)
        save_path = path#返回结果的图片名
        print(save_path)
        print(upload_path)
        return render_template('upload.html',upload_path=secure_filename(f.filename),save_path=save_path)
    # 重新返回上传界面
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")
