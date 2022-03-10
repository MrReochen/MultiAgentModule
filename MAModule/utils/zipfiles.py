import os
import zipfile

def get_zip_file(input_path, result):
    files = os.listdir(input_path)
    for f in files:
        if os.path.isdir(input_path + "/" + f):
            get_zip_file(input_path + "/" + f, result)
        else:
            result.append(input_path + "/" + f)

def zip_file_path(input_path, output_path, output_name):
    f = zipfile.ZipFile(output_path + "/" + output_name, 'w', zipfile.ZIP_DEFLATED)
    filelists = []
    get_zip_file(input_path, filelists)
    for filename in filelists:
        f.write(filename)
    f.close()