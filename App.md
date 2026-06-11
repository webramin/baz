در ادامه یک برنامه کامل تحت وب با Flask و HTML آماده کرده‌ام که تمام نیاز شما را پوشش می‌دهد. این برنامه شامل یک رابط کاربری ساده و زیبا است.

📁 ساختار پروژه

ابتدا ساختار پوشه‌های پروژه را به این صورت ایجاد کنید:

```
face_clustering_app/
│
├── app.py                 (فایل اصلی برنامه)
├── requirements.txt       (کتابخانه‌های مورد نیاز)
├── uploads/              (عکس‌های آپلود شده کاربر)
├── index/                (پوشه خروجی نهایی - نام افراد)
├── templates/
│   └── index.html        (قالب رابط کاربری)
└── static/
    └── style.css         (استایل‌های اضافی - اختیاری)
```

🐍 فایل اصلی برنامه (app.py)

```python
from flask import Flask, render_template, request, jsonify, send_file
import os
import shutil
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['INDEX_FOLDER'] = 'index'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# ایجاد پوشه‌های مورد نیاز
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)

# تنظیمات تشخیص چهره با دقت بالا
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 2

def get_face_embeddings(image_path):
    """استخراج کدهای منحصربفرد از چهره‌های موجود در یک تصویر"""
    try:
        # استخراج embeddings با دقت بالا
        embeddings = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        return embeddings
    except Exception as e:
        print(f"خطا در پردازش {image_path}: {e}")
        return []

def cluster_faces(embeddings_list):
    """خوشه‌بندی چهره‌های مشابه با استفاده از DBSCAN"""
    if not embeddings_list:
        return []
    
    # استخراج بردارهای ویژگی
    feature_vectors = []
    valid_images = []
    
    for img_path, emb_list in embeddings_list:
        for emb in emb_list:
            feature_vectors.append(emb["embedding"])
            valid_images.append((img_path, emb["face"]))
    
    if not feature_vectors:
        return []
    
    # خوشه‌بندی
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine')
    labels = clustering.fit_predict(feature_vectors)
    
    # گروه‌بندی تصاویر بر اساس خوشه
    clusters = defaultdict(list)
    for (img_path, face_region), label in zip(valid_images, labels):
        if label != -1:  # -1 به معنی نویز (چهره بدون گروه) است
            clusters[label].append({
                'path': img_path,
                'region': face_region,
                'filename': os.path.basename(img_path)
            })
    
    return list(clusters.values())

def crop_face_from_image(image_path, face_region):
    """برش چهره از تصویر اصلی برای نمایش"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        x = face_region['x']
        y = face_region['y']
        w = face_region['w']
        h = face_region['h']
        
        face_img = img[y:y+h, x:x+w]
        
        # تبدیل به base64 برای نمایش در HTML
        _, buffer = cv2.imencode('.jpg', face_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    except Exception as e:
        print(f"خطا در برش چهره: {e}")
        return None

@app.route('/')
def index():
    """صفحه اصلی برنامه"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """دریافت عکس‌ها و پردازش اولیه"""
    if 'images' not in request.files:
        return jsonify({'error': 'هیچ فایلی آپلود نشده است'}), 400
    
    files = request.files.getlist('images')
    saved_paths = []
    
    # ذخیره فایل‌های آپلود شده
    for file in files:
        if file.filename:
            filename = file.filename
            # اطمینان از یکتا بودن نام فایل
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            counter = 1
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            while os.path.exists(save_path):
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_{counter}{ext}")
                counter += 1
            
            file.save(save_path)
            saved_paths.append(save_path)
    
    if not saved_paths:
        return jsonify({'error': 'هیچ فایل معتبری آپلود نشد'}), 400
    
    # پردازش چهره‌ها
    try:
        embeddings_list = []
        for img_path in saved_paths:
            embeddings = get_face_embeddings(img_path)
            if embeddings:
                embeddings_list.append((img_path, embeddings))
        
        if not embeddings_list:
            return jsonify({'error': 'هیچ چهره‌ای در تصاویر یافت نشد'}), 400
        
        # خوشه‌بندی چهره‌ها
        clusters = cluster_faces(embeddings_list)
        
        if not clusters:
            return jsonify({'error': 'چهره کافی برای خوشه‌بندی یافت نشد'}), 400
        
        # ذخیره نتایج در session (برای استفاده در مرحله بعد)
        # در اینجا ساده‌سازی شده - در پروژه واقعی از session یا دیتابیس استفاده کنید
        
        # تبدیل خوشه‌ها به فرمت قابل ارسال
        result_clusters = []
        for idx, cluster in enumerate(clusters):
            faces_html = []
            for face in cluster[:3]:  # حداکثر 3 نمونه برای نمایش
                face_img_base64 = crop_face_from_image(face['path'], face['region'])
                if face_img_base64:
                    faces_html.append(face_img_base64)
            
            result_clusters.append({
                'id': idx,
                'count': len(cluster),
                'sample_faces': faces_html,
                'images': cluster  # ذخیره اطلاعات کامل برای مرحله بعد
            })
        
        # ذخیره موقت خوشه‌ها (در حافظه ساده شده)
        app.config['TEMP_CLUSTERS'] = result_clusters
        
        return jsonify({
            'success': True,
            'num_clusters': len(clusters),
            'clusters': result_clusters
        })
        
    except Exception as e:
        return jsonify({'error': f'خطا در پردازش: {str(e)}'}), 500

@app.route('/save_names', methods=['POST'])
def save_names():
    """ذخیره نام افراد و انتقال فایل‌ها به پوشه index/"""
    data = request.json
    cluster_names = data.get('names', {})
    temp_clusters = app.config.get('TEMP_CLUSTERS', [])
    
    if not cluster_names or not temp_clusters:
        return jsonify({'error': 'داده معتبر وجود ندارد'}), 400
    
    try:
        for cluster in temp_clusters:
            cluster_id = str(cluster['id'])
            if cluster_id in cluster_names:
                person_name = cluster_names[cluster_id].strip()
                if person_name:  # اگر کاربر نام وارد کرده باشد
                    # ایجاد پوشه شخص در index/
                    person_folder = os.path.join(app.config['INDEX_FOLDER'], person_name)
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # کپی کردن تمام عکس‌های این خوشه
                    for face in cluster['images']:
                        src_path = face['path']
                        filename = os.path.basename(src_path)
                        
                        # اطمینان از یکتا بودن نام در پوشه مقصد
                        dest_path = os.path.join(person_folder, filename)
                        counter = 1
                        base, ext = os.path.splitext(filename)
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(person_folder, f"{base}_{counter}{ext}")
                            counter += 1
                        
                        shutil.copy2(src_path, dest_path)
        
        return jsonify({
            'success': True,
            'message': 'تصاویر با موفقیت در پوشه‌های مربوطه ذخیره شدند',
            'output_folder': app.config['INDEX_FOLDER']
        })
        
    except Exception as e:
        return jsonify({'error': f'خطا در ذخیره‌سازی: {str(e)}'}), 500

@app.route('/download_output', methods=['GET'])
def download_output():
    """دانلود خروجی به صورت فایل فشرده"""
    zip_path = "output_archive.zip"
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(app.config['INDEX_FOLDER']):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, app.config['INDEX_FOLDER'])
                    zipf.write(file_path, arcname)
        
        return send_file(zip_path, as_attachment=True, download_name="classified_faces.zip")
    
    except Exception as e:
        return jsonify({'error': f'خطا در ایجاد فایل فشرده: {str(e)}'}), 500

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """پاک کردن داده‌های موقت"""
    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
        shutil.rmtree(app.config['INDEX_FOLDER'])
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)
        app.config['TEMP_CLUSTERS'] = []
        return jsonify({'success': True, 'message': 'داده‌ها با موفقیت پاک شدند'})
    except Exception as e:
        return jsonify({'error': f'خطا در پاک کردن داده‌ها: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

🎨 فایل HTML (templates/index.html)

```html
<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>سیستم تشخیص و دسته‌بندی چهره</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Tahoma', 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .content {
            padding: 30px;
        }
        
        .upload-section {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        
        .upload-section:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .file-input {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            margin: 10px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: #6c757d;
        }
        
        .btn-success {
            background: #28a745;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .clusters-container {
            margin-top: 30px;
        }
        
        .cluster-card {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
        }
        
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .cluster-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        
        .cluster-count {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        
        .face-samples {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .face-sample {
            width: 100px;
            height: 100px;
            border-radius: 10px;
            overflow: hidden;
            border: 2px solid #ddd;
        }
        
        .face-sample img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .name-input {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            font-family: inherit;
            margin-top: 10px;
        }
        
        .name-input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .action-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 30px;
            flex-wrap: wrap;
        }
        
        .file-list {
            margin-top: 15px;
            font-size: 12px;
            color: #666;
        }
        
        .alert {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        @media (max-width: 768px) {
            .content {
                padding: 15px;
            }
            
            .face-sample {
                width: 70px;
                height: 70px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 سیستم تشخیص و دسته‌بندی چهره</h1>
            <p>عکس‌های خود را آپلود کنید تا چهره‌های مشابه به صورت خودکار شناسایی و دسته‌بندی شوند</p>
        </div>
        
        <div class="content">
            <div id="alert" class="alert"></div>
            
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <h3>برای آپلود عکس‌ها کلیک کنید</h3>
                <p>می‌توانید چندین عکس را همزمان انتخاب کنید</p>
                <input type="file" id="fileInput" multiple accept="image/*" style="display: none">
                <div id="fileList" class="file-list"></div>
            </div>
            
            <div class="action-buttons">
                <button class="btn" onclick="uploadImages()" id="uploadBtn">🔍 شروع پردازش</button>
                <button class="btn btn-secondary" onclick="clearAll()" id="clearBtn">🗑️ پاک کردن همه</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>در حال پردازش چهره‌ها...</p>
                <p style="font-size: 12px; margin-top: 10px;">این عملیات ممکن است چند لحظه طول بکشد</p>
            </div>
            
            <div id="results" class="clusters-container"></div>
            
            <div class="action-buttons" id="saveActions" style="display: none;">
                <button class="btn btn-success" onclick="saveNames()">💾 ذخیره و دسته‌بندی نهایی</button>
                <button class="btn btn-secondary" onclick="downloadOutput()">📥 دانلود خروجی</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentClusters = [];
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const files = Array.from(e.target.files);
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = `<strong>${files.length} فایل انتخاب شده:</strong> ${files.map(f => f.name).join(', ')}`;
        });
        
        async function uploadImages() {
            const files = document.getElementById('fileInput').files;
            if (files.length === 0) {
                showAlert('لطفاً حداقل یک عکس انتخاب کنید', 'error');
                return;
            }
            
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('images', files[i]);
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showAlert(`${data.num_clusters} چهره متفاوت شناسایی شد! لطفاً برای هر کدام نام وارد کنید.`, 'success');
                    displayClusters(data.clusters);
                    currentClusters = data.clusters;
                    document.getElementById('saveActions').style.display = 'flex';
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('خطا در ارتباط با سرور: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function displayClusters(clusters) {
            const container = document.getElementById('results');
            container.innerHTML = '';
            
            clusters.forEach(cluster => {
                const clusterCard = document.createElement('div');
                clusterCard.className = 'cluster-card';
                clusterCard.innerHTML = `
                    <div class="cluster-header">
                        <div class="cluster-title">گروه ${cluster.id + 1}</div>
                        <div class="cluster-count">${cluster.count} عکس</div>
                    </div>
                    <div class="face-samples">
                        ${cluster.sample_faces.map(face => `
                            <div class="face-sample">
                                <img src="data:image/jpeg;base64,${face}" alt="چهره نمونه">
                            </div>
                        `).join('')}
                    </div>
                    <input type="text" class="name-input" id="name_${cluster.id}" 
                           placeholder="نام این شخص را وارد کنید (مثلاً: رضا، محمد، سارا)">
                `;
                container.appendChild(clusterCard);
            });
        }
        
        async function saveNames() {
            const names = {};
            currentClusters.forEach(cluster => {
                const input = document.getElementById(`name_${cluster.id}`);
                if (input && input.value.trim()) {
                    names[cluster.id] = input.value.trim();
                }
            });
            
            if (Object.keys(names).length === 0) {
                showAlert('لطفاً حداقل برای یک گروه نام وارد کنید', 'error');
                return;
            }
            
            showLoading(true);
            
            try {
                const response = await fetch('/save_names', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ names: names })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showAlert(`✅ ${data.message}`, 'success');
                    showAlert('می‌توانید خروجی را دانلود کنید یا عملیات جدیدی انجام دهید', 'success');
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('خطا در ذخیره‌سازی: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        async function downloadOutput() {
            try {
                window.location.href = '/download_output';
                showAlert('دانلود آغاز شد...', 'success');
            } catch (error) {
                showAlert('خطا در دانلود: ' + error.message, 'error');
            }
        }
        
        async function clearAll() {
            if (!confirm('آیا از پاک کردن همه داده‌ها مطمئن هستید؟')) return;
            
            showLoading(true);
            
            try {
                const response = await fetch('/clear_data', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showAlert('همه داده‌ها پاک شدند', 'success');
                    document.getElementById('results').innerHTML = '';
                    document.getElementById('fileInput').value = '';
                    document.getElementById('fileList').innerHTML = '';
                    document.getElementById('saveActions').style.display = 'none';
                    currentClusters = [];
                } else {
                    showAlert(data.error, 'error');
                }
            } catch (error) {
                showAlert('خطا در پاک کردن: ' + error.message, 'error');
            } finally {
                showLoading(false);
            }
        }
        
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.textContent = message;
            alert.className = `alert alert-${type}`;
            alert.style.display = 'block';
            
            setTimeout(() => {
                alert.style.display = 'none';
            }, 5000);
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
            document.getElementById('uploadBtn').disabled = show;
            document.getElementById('clearBtn').disabled = show;
        }
    </script>
</body>
</html>
```

📦 فایل requirements.txt

```txt
Flask==2.3.3
deepface==0.0.79
scikit-learn==1.3.0
opencv-python==4.8.1.78
Pillow==10.0.0
numpy==1.24.3
```

🚀 نحوه اجرا

1. ایجاد ساختار پوشه‌ها به صورت گفته شده در بالا
2. نصب کتابخانه‌ها:
   ```bash
   pip install -r requirements.txt
   ```
3. اجرای برنامه:
   ```bash
   python app.py
   ```
4. باز کردن مرورگر و رفتن به آدرس: http://localhost:5000

✨ ویژگی‌های برنامه

· رابط کاربری زیبا و واکنش‌گرا (Responsive)
· پشتیبانی از آپلود همزمان چندین عکس
· تشخیص خودکار چهره‌ها با دقت بالا (با استفاده از Facenet512 + RetinaFace)
· خوشه‌بندی هوشمند چهره‌های مشابه
· نمایش نمونه چهره‌ها برای هر گروه
· دریافت نام از کاربر برای هر شخص
· ساخت خودکار پوشه‌ها با نام افراد در پوشه index/
· قابلیت دانلود خروجی به صورت فایل فشرده
· امکان پاک کردن داده‌ها و شروع مجدد

⚙️ تنظیمات برای دقت بالاتر

در فایل app.py می‌توانید این مقادیر را تغییر دهید:

```python
DBSCAN_EPS = 0.4      # کاهش برای جداسازی بیشتر، افزایش برای گروه‌بندی بیشتر
DBSCAN_MIN_SAMPLES = 2 # حداقل تعداد چهره برای تشکیل یک گروه
```

این برنامه آماده اجرا است و تمام نیازهای شما را پوشش می‌دهد. اگر سوالی دارید یا نیاز به تغییر خاصی دارید، خوشحال می‌شوم کمک کنم!
