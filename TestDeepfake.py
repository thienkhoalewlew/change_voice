import warnings

import imageio.v2 as imageio
from skimage import img_as_ubyte
from skimage.transform import resize

from lib.deepfake.demo import load_checkpoints
from lib.deepfake.demo import make_animation

warnings.filterwarnings("ignore")

source_image = imageio.imread('content/102210212.jpg')
reader = imageio.get_reader('content/trump.mp4')

# Resize image and video to 256x256

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
generator, kp_detector = load_checkpoints(config_path='lib/config/vox-256.yaml',
                            checkpoint_path='content/models/vox-cpk.pth.tar',cpu=True)

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,cpu=True)

#save resulting video
imageio.mimsave('content/DemoKhoaGenerated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)