import gradio as gr
import csv
import librosa
import subprocess
import os

model_directory = 'models'

# Đọc dữ liệu từ file CSV
with open('output.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Đọc dòng tiêu đề
    data = [row for row in reader]  # Đọc dữ liệu

# Tách dữ liệu thành 2 danh sách: models và dl_models
dl_models = [(row[0], row[1]) for row in data]
models = os.listdir(model_directory)

# Hàm xử lý chuyển đổi file âm thanh
def convert_audio(audio_file, model_name):
    # Lưu file âm thanh đã chuyển đổi
    converted_audio = 'converted_audio.wav'
    librosa.output.write_wav(converted_audio, y, sr)

    return gr.Audio.update(value=converted_audio, interactive=False)

# Hàm cập nhật dropdowns
def update_dropdowns():
    print(models)
    model_dropdown.choices = [model for model in models]
    dl_model_dropdown.choices = [model for model in dl_models]

# Hàm download model
def download_model(model_name):
    for model in dl_models:
        if model[0] == model_name:
            print("Link for model:", model[1])  # In ra link để kiểm tra
            subprocess.run(['open', model[1]])

# Tạo giao diện
with gr.Blocks() as demo:
    gr.Markdown("Chuyển đổi file âm thanh")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Chọn file âm thanh")
            model_dropdown = gr.Dropdown(choices=models, label="Chọn mô hình")
            convert_button = gr.Button("Chuyển đổi")
        with gr.Column():
            dl_model_dropdown = gr.Dropdown(choices=[model for model in dl_models], label="Dowload Model")
            dl_btn = gr.Button("Dowload Model")
        with gr.Column():
            output_audio = gr.Audio(interactive=False, value=None, label="File âm thanh đã chuyển đổi")
            download_button = gr.Button("Tải xuống")

    # Gọi hàm cập nhật dropdowns khi giao diện được khởi tạo
    update_dropdowns()

    convert_button.click(convert_audio, inputs=[audio_input, model_dropdown], outputs=output_audio)
    dl_btn.click(download_model, inputs=[dl_model_dropdown])
demo.launch()
