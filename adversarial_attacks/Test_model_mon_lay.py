import torchvision, time, numpy as np
import eagerpy as ep, csv, os, foolbox, torch, torchvision.transforms as transforms
from torchmetrics import Accuracy
from modules.module_cif10_res50_mon_lay import Netmonogenic
import torch.nn as nn
from modules.back_ends import ResNetBackEnd, Bottleneck
from layers.monogenic import Monogenic
from utils import Utils

class TestModel:

    def __init__(self, args, ckpt_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.work_dir = os.path.abspath(os.curdir)
        self.plotDir = self.args.general_storage + '/plots'
        self.path_results = self.args.general_storage + '/cifar10_mon_performance.csv'
        self.utils = Utils()
        try:
            os.mkdir(self.plotDir)
        except:
            pass

        ### Load checkpoint for model using pytorch base
        checkpoint = torch.load(ckpt_path)
        model_weights = checkpoint["state_dict"]
        for key in list(model_weights):
            model_weights[key.replace("model.", "")] = model_weights.pop(key)
        self.model = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3], inplanes=6)
        self.model.load_state_dict(model_weights)
        self.model_test = self.model.eval().to(self.device)
        self.model_adv = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3], inplanes=3)

        self.monogenic = Monogenic()
        self.sx, self.sy, self.wl = self.monogenic.sigmax, self.monogenic.sigmay, self.monogenic.wave_length1
        self.net_mon = Netmonogenic(args)
        self.trans_mean, self.trans_std = self.net_mon.trans_mean, self.net_mon.trans_std
        self.ckpt_path = ckpt_path

        self.transform_test = transforms.Compose([torchvision.transforms.ToTensor(),
                                                 transforms.Normalize(self.trans_mean, self.trans_std)])
        self.test_set = torchvision.datasets.CIFAR10(root=str(self.args.data_path), train=False,
                                                     download=True, transform=self.transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.args.batch_size,
                                                       shuffle=False, num_workers=self.args.workers, pin_memory=True)
        self.attack_name = ''
        self.len_test = len(self.test_loader) * self.args.batch_size
        self.counter_test = 1
        self.counter_attack = 1
        self.attack_steps = 0
        self.test_examples = []
        self.test_examples_mon = []
        self.adv_test_examples = []
        self.adv_test_examples_mon = []
        self.m_attack = False
        self.m_train = False
        self.m_test = False
        self.plot_img = False
        self.monogenic_net = True
        self.attack_name = ''
        self.plot_step = 100
        self.epsilon = []
        self.attack_steps = 20

        print(f'\nBest sx: {self.sx} | sy: {self.sy} | wl: {self.wl}')

    def forward(self, x):

        with torch.no_grad():
            # -------------------- Save images for plots --------------
            sample_every = int(self.len_test / (self.args.batch_size * self.plot_step))
            step_trigger = False

            if self.plot_img:
                if self.m_test:
                    if self.plot_step > 0 and int(self.counter_test % sample_every + 1) == sample_every:
                        step_trigger = True
                        self.test_examples.append(x[:, :, :, :])
                    self.counter_test += 1

                if self.m_attack:
                    if self.plot_step > 0 and int(self.counter_attack % sample_every + 1) == sample_every:
                        step_trigger = True
                        self.adv_test_examples.append(x[:, :, :, :])
                    self.counter_attack += 1
            ### ----------------------------------------------------------

            if self.monogenic_net:
                self.monogenic.step_trigger = step_trigger

                x = self.monogenic.forward(x)

            # -------------------- Save images for plots --------------
            if self.plot_img:
                if step_trigger and self.monogenic_net:
                    if self.m_test:
                        self.test_examples_mon.append(x[:, :, :, :])

                    if self.m_attack:
                        self.adv_test_examples_mon.append(x[:, :, :, :])
            # ### ------------------------------------------------------

            outputs = self.model_test(x)

        return outputs

    def test(self, plot_img, monogenic_net):

        self.m_test, self.plot_img, self.monogenic_net = True, plot_img, monogenic_net
        i, acc_arr, acc_arr, start_time, start_datetime = 0, [], [], time.time(), self.utils.getDateTime()
        for data in self.test_loader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.forward(images)

            ## ------- Get prediction by selecting the class with the highest probability ---------
            get_acc = Accuracy(top_k=1, task='multiclass', num_classes=10).to(self.device)
            acc = get_acc(outputs, labels)
            acc_arr.append(acc.item())

            if int(i % (self.len_test / (self.args.batch_size * 10))) + 1 == int(self.len_test / (self.args.batch_size * 10)):
                print(f'Test step:{i + 1} | act. acc: {acc.item():.4f}')

            i += 1

        avg_acc = np.round(np.array(acc_arr).sum() / len(acc_arr), 4)
        task = 'clean_images_with_monogenic' if monogenic_net else 'clean_images_without_monogenic'
        print(f'\nPrediction {task}: {avg_acc}')
        self.utils.store_results(self.path_results, [f'Testng_{task}', start_datetime, self.utils.getDateTime(), 'na', 'na',
                            self.args.batch_size, 'na', 'na', 'na', 'na', avg_acc, '-',
                            f'{self.sx.item():.4f}_{self.sy.item():.4f}_{self.wl.item():.4f}', '-', self.ckpt_path])
        self.m_test = False

        return [self.test_examples, self.test_examples_mon]

    def test_attack(self, epsilon, plot_img, monogenic_net, steps=20):

        self.m_attack, self.plot_img, self.monogenic_net, self.epsilon = True, plot_img, monogenic_net, epsilon
        self.attack_step = steps

        attk_model = self.model_adv

        ## --------- Calculate bounds from transformed dataset -----------
        min_bound, max_bound = 0, 0
        for mean, std in zip(self.trans_mean, self.trans_std):
            if (0 - mean) / std < min_bound:
                min_bound = (0 - mean) / std
            if (1 - mean) / std > max_bound:
                max_bound = (1 - mean) / std

        ### ------------- Parameters of model for attack ---------------
        bounds = (min_bound, max_bound)
        fmodel = foolbox.PyTorchModel(attk_model, bounds=bounds,
                                      preprocessing=dict(mean=self.trans_mean, std=self.trans_std, axis=-3))
        fmodel = fmodel.transform_bounds(bounds)

        ### ------------- Defining attack ----------------
        epsilon, self.attack_steps = epsilon, steps
        attack = foolbox.attacks.LinfPGD(steps=steps)
        self.attack_name = self.model_test.attack_name = str(attack)[0:str(attack).find('(')]
        print(f'\nAttack: {str(attack)}\nBounds: {min_bound, max_bound} | epsilon: {epsilon}\n')

        ### ----------------- Generating Adversarial examples and testing with trained model ------------------------
        s_attack = self.utils.getDateTime()
        print(f'Len test: {self.len_test} | Batches attack: {self.len_test / self.args.batch_size} | Attack initiated at {s_attack}\n')
        att_acc = Accuracy(top_k=1, task='multiclass', num_classes=10).to(self.device)

        i, acc_arr, acc_arr_clean, tic_attack = 0, [], [], time.time()
        for data in self.test_loader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # self.model_test.m_attack = False
            ### ----------------- Create adversarial examples --------------------------
            images, labels = ep.astensor(images), ep.astensor(labels)
            raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilon)
            adv_examples, labels = clipped[0].raw, labels.raw

            # self.model_test.m_attack = True

            outputs = self.forward(adv_examples)

            ## ------- Get accuracy ---------
            acc = att_acc(outputs, labels)
            acc_arr.append(acc.item())

            ### ------------ print parameters obtained every certain step ---------------
            if int(i % (self.len_test / (self.args.batch_size * 10))) + 1 == int(self.len_test / (self.args.batch_size * 10)):
                print(f'Attack step:{i + 1} | E. time: {(time.time() - tic_attack) / 60:.3f} min | act. acc: '
                      f'{acc.item():.4f}')

            i += 1

        lapsed_t_attack, f_attack = np.round((time.time() - tic_attack) / 60, 3), self.utils.getDateTime()
        attack_avg_acc = np.round(np.array(acc_arr).sum() / len(acc_arr), 4)

        print(f'Attack finished at {f_attack} | Elapsed time: {lapsed_t_attack} | Acc: {attack_avg_acc}')

        ## -------------- Store metrics obtained from attack ------------------------
        task = 'Net_Monogenic' if monogenic_net else 'Net_without_Monogenic'
        self.utils.store_results(self.path_results, [f'Testing_adversarial_examples_through_{task}', s_attack, f_attack,
                                                lapsed_t_attack, str(attack)[0:str(attack).find('(')], epsilon,
                                                self.len_test, steps, 'na', 'na', attack_avg_acc, '-',
                                                f'{self.sx.item():.4f}_{self.sy.item():.4f}_{self.wl.item():.4f}',
                                                '-', self.ckpt_path])

        self.m_attack = False

        return [self.adv_test_examples, self.adv_test_examples_mon]


    def image_quality_measure(self, t_e, a_t_e, monogenic_net_on, plot_images_on, quality_img_on, fourier_on):

        ssim_B, ssim_B1, ssim_B2, psnr_B, psnr_B1, psnr_B2 = [], [], [], [], [], []

        for batch in range(len(t_e[0])):
            nums, i_counter = np.random.choice(len(t_e[0][batch]), 30), 0
            if batch == 8:
                nums = np.append(nums, 0)
            print(f'Img qty: {len(t_e[0][batch])}')
            for im in nums:
                if monogenic_net_on:
                    print(f'\niter image_counter: {i_counter} | Batch: {batch} | Image: {im}')

                    if len(t_e[1][batch][im]) == 6:
                        i_clean, i_clean_mon1, i_clean_mon2 = t_e[0][batch][im], t_e[1][batch][im][0:3], t_e[1][batch][im][3:6]  ### Get images for comparison
                        i_adv, i_adv_mon1, i_adv_mon2 = a_t_e[0][batch][im], a_t_e[1][batch][im][0:3], a_t_e[1][batch][im][3:6]  ### Get images for comparison
                        ### Create array of images to compare
                        clean_arr = [self.utils.preprocessimage(i_clean, unNormalize=True),
                                     self.utils.preprocessimage(i_clean_mon1), self.utils.preprocessimage(i_clean_mon2)]
                        adv_arr = [self.utils.preprocessimage(i_adv, unNormalize=True), self.utils.preprocessimage(i_adv_mon1),
                                   self.utils.preprocessimage(i_adv_mon2)]

                    elif len(t_e[1][batch][im]) == 3:
                        i_clean, i_clean_mon1 = t_e[0][batch][im], t_e[1][batch][im][:]  ### Get images for comparison
                        i_adv, i_adv_mon1 = a_t_e[0][batch][im], a_t_e[1][batch][im][:]  ### Get images for comparison
                        ### Create array of images to compare
                        clean_arr = [self.utils.preprocessimage(i_clean, unNormalize=True),
                                     self.utils.preprocessimage(i_clean_mon1)]
                        adv_arr = [self.utils.preprocessimage(i_adv, unNormalize=True), self.utils.preprocessimage(i_adv_mon1)]

                else:
                    i_clean, i_adv = t_e[0][batch][im], a_t_e[0][batch][im]  ### Get images for comparison
                    clean_arr = [
                        self.utils.preprocessimage(i_clean, unNormalize=True)]  ### Create array of images to compare
                    adv_arr = [self.utils.preprocessimage(i_adv, unNormalize=True)]  ### Create array of images to compare

                ### Get SSIM and PSNR as list
                ssim_arr, psnr_arr = self.utils.comp_met_arr(clean_arr, adv_arr)

                if plot_images_on and batch == 8 and im == 0:
                    ### Generate plot of images and values obtained from comparison
                    title = f'{self.attack_name}\n Epsilon: {self.epsilon} | sx: {self.sx.item():.4f} | sy: {self.sy.item():.4f} | ' \
                            f'wl: {self.wl.item():.4f} | Attack steps: {self.attack_steps}' if monogenic_net_on else \
                        f'{self.attack_name}\n Epsilon: {self.epsilon} | Attack steps: {self.attack_steps}'

                    self.utils.imshow_comparative(clean_arr, adv_arr, ssim_arr, psnr_arr, title, self.plotDir)

                    # self.utils.imshow2(adv_arr[0] - clean_arr[0], 'Adv. attack addition', self.plotDir)
                    # self.utils.imshow2(adv_arr[0], 'Adv. attack addition 1', self.plotDir)
                    # self.utils.imshow2(clean_arr[0], 'Adv. attack addition 2', self.plotDir)

                print(f'SSIM array: {ssim_arr} | PSNR array: {psnr_arr}')

                if quality_img_on:
                    if monogenic_net_on:

                        if len(ssim_arr) == 4:
                            ssim_B.append(np.round(ssim_arr[2], 4))
                            ssim_B1.append(np.round(ssim_arr[3], 4))
                            psnr_B.append(np.round(psnr_arr[2], 4))
                            psnr_B1.append(np.round(psnr_arr[3], 4))

                        elif len(ssim_arr) == 6:
                            ssim_B.append(np.round(ssim_arr[3], 4))
                            ssim_B1.append(np.round(ssim_arr[4], 4))
                            ssim_B2.append(np.round(ssim_arr[5], 4))
                            psnr_B.append(np.round(psnr_arr[3], 4))
                            psnr_B1.append(np.round(psnr_arr[4], 4))
                            psnr_B2.append(np.round(psnr_arr[5], 4))

                if fourier_on:

                    for i in range(len(adv_arr)):
                        clean_arr[i] = self.utils.fourier2d(torch.as_tensor(clean_arr[i]))
                        adv_arr[i] = self.utils.fourier2d(torch.as_tensor(adv_arr[i]))

                    ssim_arr, psnr_arr = self.utils.comp_met_arr(clean_arr, adv_arr)

                    if plot_images_on and batch == 8 and im == 0:
                        self.utils.imshow_comparative(clean_arr, adv_arr, ssim_arr, psnr_arr, title, self.plotDir)

                        self.utils.imshow2(clean_arr[0] - adv_arr[0], f'{title}\nSubtract fft original - fft adv. example',
                                      self.plotDir)

                    print(f'Fourier-SSIM array: {ssim_arr} | Fourier-PSNR array: {psnr_arr}')

                i_counter += 1
            print(f'size ssim B arr: {len(ssim_B)}')

        if quality_img_on:
            if monogenic_net_on:
                if len(ssim_arr) == 4:
                    ssim_B.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH3')
                    self.utils.store_results(self.args.general_storage + '/ssim_b.csv', ssim_B)
                    ssim_B1.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH3')
                    self.utils.store_results(self.args.general_storage + '/ssim_b1.csv', ssim_B1)
                    psnr_B.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH3')
                    self.utils.store_results(self.args.general_storage + '/psnr_b.csv', psnr_B)
                    psnr_B1.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH3')
                    self.utils.store_results(self.args.general_storage + '/psnr_b1.csv', psnr_B1)

                if len(ssim_arr) == 6:
                    ssim_B.append(self.epsilon)
                    self.utils.store_results(self.args.general_storage + '/ssim_b.csv', ssim_B)
                    ssim_B1.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH6')
                    self.utils.store_results(self.args.general_storage + '/ssim_b1.csv', ssim_B1)
                    ssim_B2.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH6')
                    self.utils.store_results(self.args.general_storage + '/ssim_b2.csv', ssim_B2)
                    psnr_B.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH6')
                    self.utils.store_results(self.args.general_storage + '/psnr_b.csv', psnr_B)
                    psnr_B1.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH6')
                    self.utils.store_results(self.args.general_storage + '/psnr_b1.csv', psnr_B1)
                    psnr_B2.append(f'{self.epsilon}, {self.utils.getDateTime()}, CH6')
                    self.utils.store_results(self.args.general_storage + '/psnr_b2.csv', psnr_B2)
