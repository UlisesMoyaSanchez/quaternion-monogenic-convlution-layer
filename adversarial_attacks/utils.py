import csv, datetime, pytz, matplotlib.pyplot as plt, numpy as np, time
import torch.fft
from skimage import metrics

class Utils:

    def __init__(self):
        pass

    def store_results(self, file_path, data_list_15):
        with open(file_path, 'a') as f:
            write = csv.writer(f)
            write.writerow(data_list_15)


    def getDateTime(self):
        now = datetime.datetime.now(pytz.timezone('America/Mexico_City'))

        return f'{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}'


    def imshow(self, img_o, title, plot_dir, unNormalize=False):
        npimg = self.preprocessimage(img_o, unNormalize)
        # plt.set_cmap("gray")
        plt.imshow(npimg)
        # plt.axis('off')
        plt.title(title)
        plt.savefig(f'{plot_dir}/{self.getDateTime()}')
        plt.close('all')
        time.sleep(1)

    def imshow2(self, img_o, title, plot_dir):
        npimg = img_o
        # plt.set_cmap("gray")
        plt.imshow(npimg)
        # plt.axis('off')
        plt.title(title)
        plt.savefig(f'{plot_dir}/{self.getDateTime()}')
        plt.close('all')
        time.sleep(1)

    def preprocessimage(self, img_o, unNormalize=False):
        img = img_o.detach().clone().cpu().numpy()
        if unNormalize:
            img[0], img[1], img[2] = img[0] * 0.2023 + 0.4914, img[1] * 0.1994 + 0.4822, img[2] * 0.2010 + 0.4465
        if len(img.shape) > 2:
            img = np.transpose(img, (1, 2, 0))

        return img

    def imshow_comparative(self, clean_arr, adv_arr, ssim_arr, psnr_arr, title, plot_dir):

        rows, cols = 2, len(clean_arr)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 9), sharex=True, sharey=True)
        fig.suptitle(title)
        # ax = axes.ravel()

        for i in range(rows):
            for j in range(cols):
                if cols > 1:
                    if i == 0:
                        axes[i, j].imshow(clean_arr[j])
                        # axes[i, j].axis('off')
                        axes[i, j].set_xlabel(f'SSIM: {ssim_arr[j]:.4f} | PSNR: {psnr_arr[j]:.4f}')
                    else:
                        axes[i, j].imshow(adv_arr[j])
                        # axes[i, j].axis('off')
                        axes[i, j].set_xlabel(f'SSIM: {ssim_arr[j + int(cols)]:.4f} | PSNR: {psnr_arr[j + int(cols)]:.4f}')
                else:
                    if i == 0:
                        axes[i].imshow(clean_arr[j])
                        # axes[i].axis('off')
                        axes[i].set_xlabel(f'SSIM: {ssim_arr[0]:.4f} | PSNR: {psnr_arr[0]:.4f}')
                    else:
                        axes[i].imshow(adv_arr[j])
                        # axes[i].axis('off')
                        axes[i].set_xlabel(f'SSIM: {ssim_arr[1]:.4f} | PSNR: {psnr_arr[1]:.4f}')

        plt.set_cmap("gray")
        # plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{self.getDateTime()}')
        plt.close('all')
        time.sleep(1)

    def ssim(self, imgClean, imgNoise):
        if len(imgClean.shape) > 2:
            ssim = metrics.structural_similarity(imgClean, imgNoise, data_range=imgClean.max() - imgClean.min(), channel_axis=2)
        else:
            ssim = metrics.structural_similarity(imgClean, imgNoise, data_range=imgClean.max() - imgClean.min())

        return ssim


    def psnr(self, imgClean, imgNoise):
        psnr = metrics.peak_signal_noise_ratio(imgClean, imgNoise, data_range=imgClean.max() - imgClean.min())

        return psnr

    def fourier2d(self, img):
        img = torch.mean(img, dim=2)
        # imshow(img, 'mean 1 ch', f'/opt/project/cifar10/ConvNet_Mon_cifar10/mon_storage/plots')
        ft = torch.fft.ifftshift(img)
        ft = torch.fft.fftn(ft)

        return np.log(abs(torch.fft.fftshift(ft.detach().clone().cpu()).numpy()))

    def comp_met_arr(self, cleanArr, advArr):

        ### Create array of SSIMs and PSNRs values from images compared
        ssim_arr, psnr_arr, rows = [], [], 2
        for i in range(rows):
            for j in range(len(advArr)):
                if i == 0:
                    ssim_arr.append(self.ssim(cleanArr[j], cleanArr[j]))
                    psnr_arr.append(self.psnr(cleanArr[j], cleanArr[j]))
                else:
                    ssim_arr.append(self.ssim(cleanArr[j], advArr[j]))
                    psnr_arr.append(self.psnr(cleanArr[j], advArr[j]))

        return ssim_arr, psnr_arr

