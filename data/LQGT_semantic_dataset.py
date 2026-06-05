import cv2
import numpy as np
import torch
import torch.utils.data as data

import data.util as util


class LQGTSemanticDataset(data.Dataset):
    """Read LQ/GT pairs and optional semantic masks.

    The mask directory should contain one mask per LQ image in the same sorted
    order. Masks may be grayscale or RGB images; they are converted to one
    binary-like channel in [0, 1].
    """

    def __init__(self, opt):
        super(LQGTSemanticDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.LQ_env, self.GT_env = None, None

        self.paths_LQ, self.sizes_LQ = util.get_image_paths(
            self.data_type, opt['dataroot_LQ'])
        self.paths_GT, self.sizes_GT = util.get_image_paths(
            self.data_type, opt.get('dataroot_GT'))
        self.paths_mask, _ = util.get_image_paths(
            self.data_type, opt.get('dataroot_mask'))

        assert self.paths_LQ, 'Error: LQ path is empty.'
        self.has_GT = self.paths_GT is not None and len(self.paths_GT) > 0
        if self.opt['phase'] == 'train':
            assert self.has_GT, 'Training requires dataroot_GT.'
        if self.has_GT:
            assert len(self.paths_LQ) == len(self.paths_GT), (
                'GT and LQ datasets have different number of images - {}, {}.'
                .format(len(self.paths_LQ), len(self.paths_GT)))
        if self.paths_mask:
            assert len(self.paths_LQ) == len(self.paths_mask), (
                'Mask and LQ datasets have different number of images - {}, {}.'
                .format(len(self.paths_mask), len(self.paths_LQ)))

    def _init_lmdb(self):
        import lmdb
        self.LQ_env = lmdb.open(
            self.opt['dataroot_LQ'], readonly=True, lock=False,
            readahead=False, meminit=False)
        if self.has_GT:
            self.GT_env = lmdb.open(
                self.opt['dataroot_GT'], readonly=True, lock=False,
                readahead=False, meminit=False)

    @staticmethod
    def _read_mask(path, size_hw, threshold):
        if path is None:
            h, w = size_hw
            return np.zeros((h, w, 1), dtype=np.float32)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError('Cannot read mask image: {}'.format(path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.float32) / 255.0
        if mask.shape[:2] != size_hw:
            mask = cv2.resize(mask, (size_hw[1], size_hw[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=2)
        if threshold is not None:
            mask = (mask >= threshold).astype(np.float32)
        return mask

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None:
            self._init_lmdb()

        scale = self.opt['scale']
        lq_path = self.paths_LQ[index]
        lq_resolution = (
            [int(s) for s in self.sizes_LQ[index].split('_')]
            if self.data_type == 'lmdb' else None)
        img_LQ = util.read_img(self.LQ_env, lq_path, lq_resolution)
        if img_LQ.shape[2] == 1:
            img_LQ = cv2.cvtColor(img_LQ, cv2.COLOR_GRAY2BGR)

        if self.has_GT:
            gt_path = self.paths_GT[index]
            gt_resolution = (
                [int(s) for s in self.sizes_GT[index].split('_')]
                if self.data_type == 'lmdb' else None)
            img_GT = util.read_img(self.GT_env, gt_path, gt_resolution)
            if img_GT.shape[2] == 1:
                img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
            if self.opt['phase'] != 'train':
                img_GT = util.modcrop(img_GT, scale)
        else:
            gt_path = lq_path
            img_GT = img_LQ.copy()

        mask_path = self.paths_mask[index] if self.paths_mask else None
        threshold = self.opt.get('mask_threshold', 0.5)
        img_mask = self._read_mask(mask_path, img_LQ.shape[:2], threshold)

        if self.opt.get('color'):
            img_GT = util.channel_convert(
                img_GT.shape[2], self.opt['color'], [img_GT])[0]
            img_LQ = util.channel_convert(
                img_LQ.shape[2], self.opt['color'], [img_LQ])[0]

        if self.opt['phase'] == 'train':
            gt_size = self.opt['GT_size']
            h, w, _ = img_LQ.shape
            lq_size = gt_size // scale
            if h < lq_size or w < lq_size:
                img_LQ = cv2.resize(
                    img_LQ, (lq_size, lq_size), interpolation=cv2.INTER_LINEAR)
                img_mask = cv2.resize(
                    img_mask, (lq_size, lq_size),
                    interpolation=cv2.INTER_NEAREST)
                if img_mask.ndim == 2:
                    img_mask = np.expand_dims(img_mask, axis=2)
                img_GT = cv2.resize(
                    img_GT, (gt_size, gt_size), interpolation=cv2.INTER_LINEAR)
                h, w, _ = img_LQ.shape

            rnd_h = np.random.randint(0, max(0, h - lq_size) + 1)
            rnd_w = np.random.randint(0, max(0, w - lq_size) + 1)
            img_LQ = img_LQ[rnd_h:rnd_h + lq_size,
                            rnd_w:rnd_w + lq_size, :]
            img_mask = img_mask[rnd_h:rnd_h + lq_size,
                                rnd_w:rnd_w + lq_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + gt_size,
                            rnd_w_GT:rnd_w_GT + gt_size, :]
            img_LQ, img_GT, img_mask = util.augment(
                [img_LQ, img_GT, img_mask],
                self.opt.get('use_flip', True),
                self.opt.get('use_rot', True))

        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_LQ, (2, 0, 1)))).float()
        img_mask = torch.from_numpy(np.ascontiguousarray(
            np.transpose(img_mask, (2, 0, 1)))).float()

        return {
            'LQ': img_LQ,
            'GT': img_GT,
            'mask': img_mask,
            'LQ_path': lq_path,
            'GT_path': gt_path,
            'mask_path': mask_path
        }

    def __len__(self):
        return len(self.paths_LQ)
