"""Contracts needed to keep pretrained geometry intact while learning features."""
import copy
import torch
from src.model.encoder.common.gmae import Transformer
from src.model.vae_feature_lifting import FrozenGeometryFeatureDecoder


def setup_decoder():
    torch.manual_seed(4)
    geometry=Transformer(dim=32,depth=2,heads=4,dim_head=8,mlp_dim=64).eval()
    decoder=FrozenGeometryFeatureDecoder(geometry,dim=32,channels=16)
    decoder.context_feature=torch.randn(1,7,32)
    decoder.gaussian_slots=torch.randn(1,3,32)
    return geometry,decoder,torch.randn(1,10,32)


def test_geometry_is_preserved_and_only_feature_parameters_receive_gradients():
    geometry,decoder,x=setup_decoder()
    with torch.no_grad():expected=geometry(x)
    actual=decoder(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert decoder.features[0].shape==(1,3,16)
    decoder.features[0].square().mean().backward()
    assert all(p.grad is None for p in geometry.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.trainable_parameters())
    assert sum(float(p.grad.abs().sum()) for p in decoder.trainable_parameters())>0


def test_feature_checkpoint_roundtrip_excludes_geometry():
    _,decoder,x=setup_decoder();decoder(x)
    expected=decoder.features[-1].detach().clone()
    state=copy.deepcopy(decoder.feature_state_dict())
    assert not any(k.startswith('geometry.') for k in state)
    with torch.no_grad():
        for p in decoder.trainable_parameters():p.add_(2)
    decoder.load_feature_state_dict(state);decoder.features=[];decoder(x)
    torch.testing.assert_close(decoder.features[-1],expected,rtol=0,atol=0)


def test_mismatched_patch_and_feature_grids_are_rejected():
    _,decoder,x=setup_decoder();decoder.context_feature=decoder.context_feature[:,:6]
    try:decoder(x)
    except ValueError as exc:assert 'Patch/feature sequence mismatch' in str(exc)
    else:raise AssertionError('Mismatched token grids must fail')
