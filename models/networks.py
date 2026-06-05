"""Network factory.

The original file imported every historical baseline at module import time.
Several of those source folders are not present in this repository, so even
valid configurations failed before model selection happened. Keep imports lazy
and report missing architectures only when they are requested.
"""


def _get(opt_net, key, default=None):
    value = opt_net.get(key, default)
    return default if value is None else value


def _missing(which_model):
    missing = [
        'JDNet', 'MPRNet', 'RIDNet', 'RLNet', 'VRGNet', 'SPANet', 'DuRN',
        'DCSFN', 'DCSFN_visual', 'JDNet_event', 'MPRNet_event',
        'RIDNet_event', 'RLNet_event', 'VRGNet_event', 'SPANet_event',
        'DuRN_event', 'DCSFN_event', 'ESTIL', 'ESTIL_event',
        'eSL_Net_deblur', 'EVDI', 'MemDeblur', 'PAN_deblur', 'STRA1',
        'D2HNet', 'D2Net', 'EFNet', 'ERDN', 'RED_Net', 'STFAN',
        'STRAHN_deblur', 'UEVD', 'LEDVI'
    ]
    if which_model in missing:
        raise NotImplementedError(
            'Generator model [{}] is referenced by an option file, but its '
            'source architecture is not included as an importable package in '
            'this repository. Use one of the registered models in README.md '
            'or add the missing source folder before running this config.'
            .format(which_model))
    raise NotImplementedError(
        'Generator model [{}] is not recognized.'.format(which_model))


def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'SemanticAwareDenoiser':
        from models.archs.semantic_aware_denoiser import SemanticAwareDenoiser
        return SemanticAwareDenoiser(
            in_nc=_get(opt_net, 'in_nc', 3),
            mask_nc=_get(opt_net, 'mask_nc', 1),
            out_nc=_get(opt_net, 'out_nc', 3),
            nf=_get(opt_net, 'nf', 48),
            num_blocks=_get(opt_net, 'num_blocks', 4),
            num_sf_blocks=_get(opt_net, 'num_sf_blocks', 3),
            patch_grid=_get(opt_net, 'patch_grid', 8))

    if which_model == 'PAN':
        import models.archs.PAN_arch as PAN_arch
        return PAN_arch.PAN(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'PAN_Event1':
        import models.archs.PAN_event1_arch as PAN_event1_arch
        return PAN_event1_arch.PAN_Event_1(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'PAN_Event2':
        import models.archs.PAN_event2_arch as PAN_event2_arch
        return PAN_event2_arch.PAN_Event_2(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'PAN_Event3':
        import models.archs.PAN_event3_arch as PAN_event3_arch
        return PAN_event3_arch.PAN_Event_3(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'PAN_Event4':
        import models.archs.PAN_event4_arch as PAN_event4_arch
        return PAN_event4_arch.PAN_Event_4(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'PAN_Event5':
        import models.archs.PAN_event5_arch as PAN_event5_arch
        return PAN_event5_arch.PAN_Event_5(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], unf=opt_net['unf'], nb=opt_net['nb'],
            scale=opt_net['scale'])

    if which_model == 'eSL':
        import models.archs.eSL as eSL
        return eSL.eSL_Net(scale=opt_net['scale'])

    if which_model == 'e2sri':
        import models.archs.e2sri as e2sri
        return e2sri.SRNet(
            scale=opt_net['scale'],
            base1_channels=_get(opt_net, 'base1_channels', 16),
            base2_channels=_get(opt_net, 'base2_channels', 32))

    if which_model == 'DCSR':
        import models.archs.dcsr as dcsr
        return dcsr.DCSR(scale=opt_net['scale'], n_feats=opt_net['n_feats'])

    if which_model == 'TDAN':
        import models.archs.TDAN_model as TDAN
        return TDAN.TDAN_VSR()

    if which_model == 'DPT':
        import models.archs.DPT as DPT
        return DPT.DPT_Net(angRes=_get(opt_net, 'angRes', 5),
                           factor=opt_net['scale'])

    if which_model == 'SPADE':
        import models.archs.spade_e2v as SPADE
        return SPADE.Unet6(scale=opt_net['scale'])

    if which_model == 'MSRResNet_PA':
        import models.archs.SRResNet_arch as SRResNet_arch
        return SRResNet_arch.MSRResNet_PA(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])

    if which_model == 'RCAN_PA':
        import models.archs.RCAN_arch as RCAN_arch
        return RCAN_arch.RCAN_PA(
            n_resgroups=opt_net['n_resgroups'],
            n_resblocks=opt_net['n_resblocks'],
            n_feats=opt_net['n_feats'],
            res_scale=opt_net['res_scale'],
            n_colors=opt_net['n_colors'],
            rgb_range=opt_net['rgb_range'],
            scale=opt_net['scale'])

    _missing(which_model)
