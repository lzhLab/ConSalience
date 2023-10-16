import os
import h5py
import torch
import numpy as np

from networks.square_wavelet_module import Wavelet_square_FPN


def wavelet_handle_x(x, input_size=512):
    x_over0 = np.where(x > 0, 1, 0)
    count_over0 = np.sum(x_over0)
    if count_over0 <= 0:
        return torch.tensor(
            np.array([np.stack([np.zeros((input_size, input_size), dtype=np.float32)])])
        )
    cal_mean = np.sum(x) / count_over0
    sqrt_x = x * x
    cal_sqrt = np.sum(sqrt_x) / count_over0
    std = abs(cal_sqrt - cal_mean**2) ** 0.5
    if std <= 0:
        return torch.tensor(
            np.array([np.stack([np.zeros((input_size, input_size), dtype=np.float32)])])
        )
    x = (x - cal_mean) / (std)
    x = x * x_over0
    return torch.tensor(np.array([np.stack([x])]), dtype=torch.float32)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavelet_model = Wavelet_square_FPN()
    wavelet_model.load_state_dict(
        torch.load("./networks/init_model/dice_best_Wavelet_square_net_vessel_data_size512.pth")
    )
    wavelet_model.to(device)
    wavelet_model.eval()

    img_dir = "./data/3D-IRCADb/vol_data"
    sa_dir = "./data/3D-IRCADb/vol_salience"
    if not os.path.exists(sa_dir):
        os.mkdir(sa_dir)

    for file in sorted(os.listdir(img_dir)):
        print(file)
        img_path = os.path.join(img_dir, file)
        sa_path = os.path.join(sa_dir, file)

        with h5py.File(img_path) as f:
            img = f["image"][:]

        # pad the sample
        INPUT_SIZE = 512
        h, w, d = img.shape
        if h < INPUT_SIZE or w < INPUT_SIZE or d < INPUT_SIZE:
            ph, pw, pd = INPUT_SIZE - h, INPUT_SIZE - w, INPUT_SIZE - d
            pad_img = np.pad(
                img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2))
            )

        # Coronal salience
        coronal_wavelet = np.zeros_like(pad_img)
        for i in range(INPUT_SIZE):
            input = wavelet_handle_x(pad_img[i, :, :]).to(device)
            output = wavelet_model(input)
            output = output.squeeze().cpu().detach().numpy()
            coronal_wavelet[i, :, :] = output
        coronal_wavelet = coronal_wavelet[
            ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w, pd // 2 : pd // 2 + d
        ]

        # Sagittal salience
        sagittal_wavelet = np.zeros_like(pad_img)
        for i in range(INPUT_SIZE):
            input = wavelet_handle_x(pad_img[:, i, :]).to(device)
            output = wavelet_model(input)
            output = output.squeeze().cpu().detach().numpy()
            sagittal_wavelet[:, i, :] = output
        sagittal_wavelet = sagittal_wavelet[
            ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w, pd // 2 : pd // 2 + d
        ]

        # Transverse salience
        transverse_wavelet = np.zeros_like(pad_img)
        for i in range(INPUT_SIZE):
            input = wavelet_handle_x(pad_img[:, :, i]).to(device)
            output = wavelet_model(input)
            output = output.squeeze().cpu().detach().numpy()
            transverse_wavelet[:, :, i] = output
        transverse_wavelet = transverse_wavelet[
            ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w, pd // 2 : pd // 2 + d
        ]

        # Concatenate the 3 saliences
        salience = np.stack((transverse_wavelet, coronal_wavelet, sagittal_wavelet), axis=0)
        print(salience.shape)

        # Normalization
        MAX = salience.max()
        MIN = salience.min()
        if MAX != MIN:
            salience = (salience - MIN) / (MAX - MIN)

        with h5py.File(sa_path, "w") as f:
            f.create_dataset(name="salience", data=salience, compression="gzip")

