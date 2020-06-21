import os 
from flask import Flask, send_file, send_from_directory, safe_join, abort
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

app.config["CLIENT_CSV"] = os.path.join(app.root_path, 'src/static/client/csv')
app.config["CLIENT_IMAGES"] = os.path.join(app.root_path, 'src/static/client/img')
app.config["CLIENT_PDF"] = os.path.join(app.root_path, 'src/static/client/pdf')
app.config["CLIENT_REPORT"] = os.path.join(app.root_path, 'src/static/reports')

@app.route("/get-image/<image_name>")
def get_image(image_name):
  print(app.config["CLIENT_IMAGES"])
  try:
    return send_from_directory(app.config["CLIENT_IMAGES"], filename=image_name, as_attachment=True)
  except FileNotFoundError:
    abort(404)

@app.route("/get-csv/<csv_id>")
def get_csv(csv_id):
  filename = f"{csv_id}.csv"

  try:
    return send_from_directory(app.config["CLIENT_CSV"], filename=filename, as_attachment=True)
  except FileNotFoundError:
    abort(404)

@app.route("/get-pdf/<pdf_id>")
def get_pdf(pdf_id):
  filename = f"{pdf_id}.pdf"

  try:
    return send_from_directory(app.config["CLIENT_PDF"], filename=filename, as_attachment=True)
  except FileNotFoundError:
    abort(404)

@app.route("/get-report/<path:path>")
def get_report(path):
  try:
    return send_from_directory(app.config["CLIENT_REPORT"], filename=path, as_attachment=True)
  except FileNotFoundError:
    abort(404)

def load_image_array():
  image = Image.open(os.path.join(app.config["CLIENT_IMAGES"], '003.jpg'))

  # we resize and rescale as if we were working with the image for ML
  image = image.resize((512, 512), Image.ANTIALIAS)
  image_array = np.asarray(image)
  rescaled = (image_array.astype('float32') - 127.5) / 127.5

  return rescaled

@app.route('/get-numpy-image')
def get_numpy_image():

    # load numpy array
    arr = load_image_array()
    arr = (arr * 127.5) + 127.5

    # convert numpy array to PIL Image
    img = Image.fromarray(arr.astype('uint8'))

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'JPEG', quality=95, optimize=True, progressive=True)

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return send_file(file_object, mimetype='image/jpeg')


if __name__ == "__main__":
  app.run(debug=True)
