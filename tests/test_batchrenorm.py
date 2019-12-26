import batchrenorm
import pytest
import torch


@pytest.mark.parametrize("kwargs", [{"affine": True}, {"affine": False}])
def test_step1(kwargs):
    for _ in range(100):
        batch_renorm = batchrenorm.BatchRenorm1d(10, **kwargs)
        x = torch.randn(100, 10)
        assert (
            batch_renorm(x)
            - torch.nn.functional.batch_norm(
                x,
                torch.zeros(10),
                torch.ones(10),
                eps=batch_renorm.eps,
                training=True,
            )
        ).abs().max() < 1e-5


@pytest.mark.parametrize("kwargs", [{"affine": True}, {"affine": False}])
@pytest.mark.seed_rng()
def test_ablation(kwargs):
    br = batchrenorm.BatchRenorm1d(5, eps=0.0, **kwargs)
    xs = torch.randn(10, 5, 5)
    xs_mean = xs.mean((0, 1))
    xs_var = xs.var((0, 1), unbiased=False)

    def _step():
        return (
            torch.stack(
                [
                    br(x)
                    - torch.nn.functional.batch_norm(
                        x, xs_mean, xs_var, eps=0.0, training=False
                    )
                    for x in xs
                ]
            )
            .abs()
            .mean()
        )

    errors = torch.stack([_step() for _ in range(100)])
    assert errors[-10:].mean() < errors[:10].mean()


def test_batchnorm1d():
    br = batchrenorm.BatchRenorm1d(3).eval()
    x = torch.randn(5, 3)
    assert (br(x) == br(x.unsqueeze(-1)).squeeze(-1)).all()
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        br(x[0])
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        br(x[..., None, None])
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        br(x[..., None, None, None])


def test_batchnorm2d():
    br = batchrenorm.BatchRenorm2d(3).eval()
    x = torch.randn(5, 3, 10, 10)
    br(x)
    assert (
        br(x[:, :, :1, :1]).squeeze()
        == batchrenorm.BatchRenorm1d(3).eval()(x[:, :, 0, 0])
    ).all()
    with pytest.raises(ValueError, match="expected 4D input"):
        br(x[0, :, 0, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        br(x[:, :, 0, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        br(x[:, :, :, 0])
    with pytest.raises(ValueError, match="expected 4D input"):
        br(x[:, :, None])


def test_batchnorm3d():
    br = batchrenorm.BatchRenorm3d(3).eval()
    x = torch.randn(5, 3, 10, 10, 10)
    br(x)
    assert (
        br(x[:, :, :1]).squeeze()
        == batchrenorm.BatchRenorm2d(3).eval()(x[:, :, 0])
    ).all()
    with pytest.raises(ValueError, match="expected 5D input"):
        br(x[0, :, 0, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        br(x[:, :, 0, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        br(x[:, :, :, 0, 0])
    with pytest.raises(ValueError, match="expected 5D input"):
        br(x[:, :, 0, :, :])
