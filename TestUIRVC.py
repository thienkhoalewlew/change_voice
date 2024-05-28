import os
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import glob
import re
import csv
import requests
import zipfile
import shutil
from scipy.io import wavfile
import moviepy.editor as mp
import matplotlib
import matplotlib.pyplot as plt

import imageio.v2 as imageio
from skimage import img_as_ubyte
from skimage.transform import resize

from lib.deepfake.demo import load_checkpoints
from lib.deepfake.demo import make_animation

from datetime import datetime
from sympy import true, false

logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info('Started')

from lib.config.config import Config
from lib.vc.vc_infer_pipeline import VC
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.vc.utils import (
    load_hubert
)

# auto load stuffs
config = Config()
spaces = os.getenv("SYSTEM") == "spaces"
force_support = None
if config.unsupported is False:
    if config.device == "mps" or config.device == "cpu":
        force_support = False
else:
    force_support = True
audio_mode = []
f0method_mode = []
f0method_info = ""
hubert_model = load_hubert(config)
model_name = ""

# Đọc dữ liệu từ file CSV vào danh sách
model_data = []
with open('output.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Bỏ qua dòng tiêu đề
    for row in reader:
        model_data.append(row)

audio_output_path = None
def run_convert(
        vc_upload,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        source_image_upload,
        reader_upload,
):
    try:
        logs = []
        logger.info(f"Converting ...")
        logs.append(f"Converting ...")
        yield "\n".join(logs), None, None,None  # Thêm None cho kết quả hình ảnh

        if vc_upload is None:
            return "You need to upload an audio", None, None

        if source_image_upload is None:
            return "You need to upload a source image", None, None

        if reader_upload is None:
            return "You need to upload a reader file", None, None

        # Xử lý chuyển đổi âm thanh
        sampling_rate, audio = vc_upload
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        times = [0, 0, 0]
        f0_up_key = int(f0_up_key)
        vc_input = os.path.join("output", "tts", "tts.mp3")
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            vc_input,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            f0_file=None,
        )

        # Xử lý chuyển đổi hình ảnh
        source_image = source_image_upload
        reader = imageio.get_reader(reader_upload.name)
        source_image = resize(source_image, (256, 256))[..., :3]
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=False)
        imageio.mimsave('content/generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

        # Lưu file âm thanh sử dụng librosa
        global audio_output_path
        audio_output_path = "content/converted_audio.wav"
        wavfile.write(audio_output_path, tgt_sr, audio_opt)

        #tttttttttttttttttttttttt
        # Vẽ phổ âm thanh trước xử lý
        fft_data_before = np.fft.rfft(audio)
        freq_before = np.fft.rfftfreq(len(audio), d=1.0 / sampling_rate)

        # Vẽ phổ âm thanh sau xử lý
        fft_data_after = np.fft.rfft(audio_opt)
        freq_after = np.fft.rfftfreq(len(audio_opt), d=1.0 / tgt_sr)

        # Tạo trục tần số mới để đồng bộ hóa hai đồ thị
        new_freq = np.linspace(0, max(freq_before.max(), freq_after.max()), 1000)
        amp_before = np.interp(new_freq, freq_before, np.abs(fft_data_before))
        amp_after = np.interp(new_freq, freq_after, np.abs(fft_data_after))

        # Vẽ đồ thị
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))  # Tạo hai đồ thị trên cùng một hình ảnh

        ax1.plot(new_freq, amp_before)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio Spectrum Before')

        ax2.plot(new_freq, amp_after)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Audio Spectrum After')

        # Lưu hình ảnh phổ vào file
        spectrum_path = "content/audio_spectrum.png"
        fig.savefig(spectrum_path)
        #tttttttttttttttttttttttt
        info = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
        logger.info(f" | {info}")
        logs.append(f"Successfully Convert \n{info}")
        yield "\n".join(logs), audio_output_path, "content/generated.mp4",spectrum_path   # Trả về đường dẫn file âm thanh và video riêng biệt
    except Exception as err:
        info = traceback.format_exc()
        logger.error(info)
        logger.error(f"Error when using .\n{str(err)}")
        yield info, None, None, None
    return run_convert

def load_model():
    global tgt_sr, net_g, vc, hubert_model, version, if_f0, file_index
    path = os.path.join("models", model_name)
    for pth_file in glob.glob(f"{path}/*.pth"):
        file_index = glob.glob(f"{path}/*.index")[0]
        if not file_index:
            logger.warning("No Index file detected!")
        cpt = torch.load(pth_file, map_location=torch.device('cpu'))
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        elif version == "v2":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
        del net_g.enc_q
        logger.info(net_g.load_state_dict(cpt["weight"], strict=False))
        net_g.eval().to(config.device)
        if config.is_half:
            net_g = net_g.half()
        else:
           net_g = net_g.float()
        vc = VC(tgt_sr, config)
    print(model_name)
    return model_name

def change_model_name(tobechangeto):
    global model_name
    model_name = tobechangeto
    load_model()
    return tobechangeto

def download_model(model):
    model_directory = 'models'
    print(model)
    parts = model.split("'")
    link = parts[3]
    print(link)
    response = requests.get(link)
    if response.status_code == 200:
        valid_file_name = re.sub(r'[<>:"/\\|?*]', "", parts[1])
        zip_file_path = f"{model_directory}/{valid_file_name}.zip"  # Đặt tên cho tệp zip
        extract_directory = os.path.join(model_directory, valid_file_name)  # Thư mục để giải nén
        os.makedirs(extract_directory, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        with open(zip_file_path, "wb") as file:
            file.write(response.content)
        if os.path.exists(extract_directory):
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_directory)  # Giải nén tất cả các tệp vào thư mục
            os.remove(zip_file_path)  # Xóa tệp zip sau khi giải nén

            # Di chuyển tệp từ thư mục con ra thư mục cha nếu có
            sub_dirs = [os.path.join(extract_directory, d) for d in os.listdir(extract_directory) if
                        os.path.isdir(os.path.join(extract_directory, d))]
            if sub_dirs:
                for sub_dir in sub_dirs:
                    for item in os.listdir(sub_dir):
                        src_path = os.path.join(sub_dir, item)
                        dest_path = os.path.join(extract_directory, item)
                        shutil.move(src_path, dest_path)
                    shutil.rmtree(sub_dir)

            print("Download successful!")
            delete_model(parts[1])
            model_dirs = get_model_dirs()
            return gr.Dropdown.update(choices=model_dirs)
        else:
            print("Failed to create extract directory.")
    else:
        print("Failed to download model.")

def delete_model(model_name):
    for i, model in enumerate(model_data):
        if model[0] == model_name:
            del model_data[i]

def get_model_dirs():
    return [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d))]

def detach_audio_from_video(video_file):
    video = mp.VideoFileClip(video_file)
    audio = video.audio
    if audio:

        audio_path = "content/detached_audio.wav"
        video_path = "content/detached_video.mp4"
        audio.write_audiofile(audio_path)
        video.without_audio().write_videofile(video_path, codec='libx264')

        return audio_path, video_path
    else:
        return None, None

# Hàm mới để ghép âm thanh và video
def combine_audio_video( video_file):
    global audio_output_path

    # Kiểm tra xem đường dẫn có phải là chuỗi hay không
    if not isinstance(audio_output_path, str) or not isinstance(video_file, str):
        return None  # Hoặc giá trị mặc định nào đó

    audio = mp.AudioFileClip(audio_output_path)
    video = mp.VideoFileClip(video_file)

    combined_clip = video.set_audio(audio)
    output_path = "content/combined_output.mp4"
    combined_clip.write_videofile(output_path)

    return output_path

if __name__ == '__main__':
    with gr.Blocks() as main_app:
        generator, kp_detector = load_checkpoints(config_path='lib/config/vox-256.yaml',
                                                  checkpoint_path='content/models/vox-cpk.pth.tar',
                                                  cpu=False)
        with gr.Row():
            dl_model_dropdown = gr.Dropdown(choices=[model for model in model_data], label="Download Model")
            dl_btn = gr.Button("Download Model")

        with gr.Tab("AI cover"):
            model_dirs = get_model_dirs()
            video_upload = gr.Video(label="Upload video file", interactive=True)
            detach_btn = gr.Button("Detach Audio and Video")
            vc_upload = gr.Audio(label="Detached audio file", interactive=False)
            reader_upload = gr.File(label="Detached video file", interactive=False)
            models_dropdown = gr.Dropdown(choices=model_dirs, label="Chọn model:")
            models_dropdown.change(fn=change_model_name, inputs=models_dropdown)
            f0_up_key = gr.Number(label="Transpose(f0_up_key)", interactive=True)
            f0_method = gr.Radio(label="Pitch extraction algorithm(f0_method)", choices=["rmvpe", "pm"], value="rmvpe", interactive=True)
            index_rate = gr.Slider(minimum=0, maximum=1, value=0.7, label="Retrieval feature ratio(index_rate)", interactive=True)
            filter_radius = gr.Slider(minimum=0, maximum=7, value=3, step=1, label="Apply Median Filtering(filter_radius)", interactive=True)
            resample_sr = gr.Slider(minimum=0, maximum=48000, value=0, label="Resampling(resample_sr)", interactive=True)
            rms_mix_rate = gr.Slider(minimum=0, maximum=1, value=1, label="Volume Envelope(rms_mix_rate)", interactive=True)
            protect = gr.Slider(minimum=0, maximum=0.5, value=0.5, label="Voice Protection(protect)", interactive=True)
            source_image_upload = gr.Image(label="Upload source image", interactive=True)
            Convertbtn = gr.Button("Convert", variant="primary")
            vc_log = gr.Textbox(label="Output Information", visible=True, interactive=False)
            vc_output = gr.Audio(label="Output Audio", interactive=False)
            output_image = gr.Video(label="Output video file", interactive=False)
            spectrograms = gr.Image(label="Spectrogram after", interactive=False)
            Convertbtn.click(fn=run_convert,
                             inputs=[vc_upload, f0_up_key, f0_method, index_rate, filter_radius, resample_sr, rms_mix_rate, protect, source_image_upload, reader_upload],
                             outputs=[vc_log, vc_output, output_image,spectrograms]
                             )
            detach_btn.click(
                fn=detach_audio_from_video,
                inputs=[video_upload],
                outputs=[vc_upload, reader_upload]
            )
            dl_btn.click(download_model, inputs=[dl_model_dropdown], outputs=models_dropdown)

            # Thêm nút ghép âm thanh và video
            combine_btn = gr.Button("Combine Audio and Video")
            combine_btn.click(
                fn=combine_audio_video,
                inputs=[ output_image],
                outputs=[gr.Video(label="Combined Output")]
            )

        main_app.queue(
            max_size=20,
            api_open=config.api,
        ).launch(
            share=False,
            max_threads=1,
            allowed_paths=["models"]
        )