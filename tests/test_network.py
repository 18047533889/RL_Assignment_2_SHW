"""
``network.py``: residual blocks, ``SuperTTTNet`` shapes/value range, gradients, masked softmax, optional Apple MPS.

No RLlib; validates CNN matches default ``config`` layout.
"""
import torch
import pytest
from network import SuperTTTNet, ResidualBlock, get_device


class TestResidualBlock:
    """``ResidualBlock``: channel shape preserved; zero weights approximate identity path."""

    def test_output_shape_preserved(self):
        """Spatial size stays 12×12."""
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 12, 12)
        out = block(x)
        assert out.shape == (2, 64, 12, 12)

    def test_skip_connection_works(self):
        """With conv weights zeroed, block reduces to ReLU(x) (tests residual skip)."""
        block = ResidualBlock(32)
        x = torch.randn(1, 32, 12, 12)
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        out = block(x)
        assert torch.allclose(out, torch.relu(x), atol=1e-5)


class TestSuperTTTNet:
    """Full net: 96 logits, scalar value, defaults, masked probability mass."""

    @pytest.fixture
    def net(self):
        return SuperTTTNet(in_channels=7, num_filters=64, num_res_blocks=2, num_actions=96)

    def test_output_shapes(self, net):
        """Forward: logits [B,96], value [B]."""
        x = torch.randn(4, 7, 12, 12)
        logits, value = net(x)
        assert logits.shape == (4, 96)
        assert value.shape == (4,)

    def test_value_in_range(self, net):
        """Value head in [-1,1] (tanh)."""
        x = torch.randn(8, 7, 12, 12)
        with torch.no_grad():
            _, value = net(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_single_sample(self, net):
        """Batch size 1 shapes."""
        x = torch.randn(1, 7, 12, 12)
        logits, value = net(x)
        assert logits.shape == (1, 96)
        assert value.shape == (1,)

    def test_gradient_flows(self, net):
        """Scalar loss backprops to all trainable params."""
        x = torch.randn(2, 7, 12, 12)
        logits, value = net(x)
        loss = logits.sum() + value.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_default_config(self):
        """Default constructor matches config depth/width."""
        net = SuperTTTNet()
        assert net.num_actions == 96
        x = torch.randn(1, 7, 12, 12)
        logits, value = net(x)
        assert logits.shape == (1, 96)

    def test_masking_zeroes_illegal_actions(self, net):
        """After -inf mask, softmax mass only on legal cells."""
        x = torch.randn(1, 7, 12, 12)
        with torch.no_grad():
            logits, _ = net(x)
        mask = torch.zeros(96)
        mask[52] = 1
        mask[53] = 1
        mask[54] = 1
        inf_mask = torch.where(mask > 0, torch.zeros_like(logits),
                               torch.full_like(logits, -1e10))
        masked = logits + inf_mask
        probs = torch.softmax(masked, dim=-1)
        assert probs[0, 0].item() < 1e-8
        assert probs[0, 52].item() > 0


class TestNetworkParameterCount:
    """Parameter counts: default large net vs small config in expected ranges."""

    def test_parameter_count_reasonable(self):
        """Default 128 filters × 6 blocks: total params in a sane band."""
        net = SuperTTTNet(in_channels=7, num_filters=128, num_res_blocks=6, num_actions=96, value_fc_hidden=512)
        total = sum(p.numel() for p in net.parameters())
        assert 500_000 < total < 8_000_000

    def test_small_config(self):
        """Small model has far fewer parameters than default."""
        net = SuperTTTNet(in_channels=7, num_filters=32, num_res_blocks=1, num_actions=96)
        total = sum(p.numel() for p in net.parameters())
        assert total < 200_000


class TestMPSDevice:
    """Apple MPS: optional forward/backward near CPU when available."""

    def test_get_device_returns_valid(self):
        """``get_device()`` returns cpu, cuda, or mps."""
        device = get_device()
        assert device.type in ("mps", "cuda", "cpu")

    @pytest.mark.skipif(not torch.backends.mps.is_available(),
                        reason="MPS not available")
    def test_forward_on_mps(self):
        """MPS forward: device and shapes."""
        device = torch.device("mps")
        net = SuperTTTNet(in_channels=7, num_filters=32, num_res_blocks=1, num_actions=96).to(device)
        x = torch.randn(2, 7, 12, 12, device=device)
        with torch.no_grad():
            logits, value = net(x)
        assert logits.device.type == "mps"
        assert value.device.type == "mps"
        assert logits.shape == (2, 96)

    @pytest.mark.skipif(not torch.backends.mps.is_available(),
                        reason="MPS not available")
    def test_gradient_on_mps(self):
        """Gradients live on MPS tensors."""
        device = torch.device("mps")
        net = SuperTTTNet(in_channels=7, num_filters=32, num_res_blocks=1, num_actions=96).to(device)
        x = torch.randn(2, 7, 12, 12, device=device)
        logits, value = net(x)
        loss = logits.sum() + value.sum()
        loss.backward()
        for p in net.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert p.grad.device.type == "mps"

    @pytest.mark.skipif(not torch.backends.mps.is_available(),
                        reason="MPS not available")
    def test_mps_vs_cpu_consistency(self):
        """Same weights: CPU vs MPS forward within tolerance."""
        torch.manual_seed(42)
        net_cpu = SuperTTTNet(in_channels=7, num_filters=32, num_res_blocks=1, num_actions=96)
        net_mps = SuperTTTNet(in_channels=7, num_filters=32, num_res_blocks=1, num_actions=96).to("mps")
        net_mps.load_state_dict(net_cpu.state_dict())
        x_cpu = torch.randn(1, 7, 12, 12)
        x_mps = x_cpu.to("mps")
        with torch.no_grad():
            logits_cpu, val_cpu = net_cpu(x_cpu)
            logits_mps, val_mps = net_mps(x_mps)
        assert torch.allclose(logits_cpu, logits_mps.cpu(), atol=1e-4)
        assert torch.allclose(val_cpu, val_mps.cpu(), atol=1e-4)
