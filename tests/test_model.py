import torch

from microllm.config import GPTConfig, preset
from microllm.data import get_batch
from microllm.generate import sample
from microllm.model import GPT
from microllm.train import lr_at

torch.manual_seed(0)


def tiny(vocab=50):
    return GPT(preset("tiny", vocab))


def test_shapes_and_loss():
    m = tiny()
    x = torch.randint(0, 50, (3, 17))
    y = torch.randint(0, 50, (3, 17))   # независимые цели: не подсказываем модели ответ входом
    logits, loss = m(x, y)
    assert logits.shape == (3, 17, 50)
    # у необученной модели ошибка ≈ ln(vocab): она «угадывает» наугад
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_causality_future_does_not_leak():
    """Меняем последний токен — предсказания для предыдущих позиций не должны измениться."""
    m = tiny().eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 50
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-5)
    assert not torch.allclose(la[:, -1], lb[:, -1])


def test_attention_maps_are_distributions_and_causal():
    m = tiny().eval()
    x = torch.randint(0, 50, (1, 10))
    _, _, attns = m(x, return_attn=True)
    assert len(attns) == m.cfg.n_layer
    a = attns[0]
    assert a.shape == (1, m.cfg.n_head, 10, 10)
    assert torch.allclose(a.sum(-1), torch.ones(1, m.cfg.n_head, 10), atol=1e-5)  # строки = вероятности
    assert torch.all(torch.triu(a[0, 0], diagonal=1) == 0)                          # будущего не видит


def test_context_too_long():
    m = tiny()
    try:
        m(torch.zeros(1, m.cfg.block_size + 1, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("ожидалась ошибка про длину контекста")


def test_weight_tying():
    m = tiny()
    assert m.head.weight.data_ptr() == m.tok_emb.weight.data_ptr()


def test_can_overfit_tiny_batch():
    """Если модель не может выучить одну маленькую пачку наизусть — в обучении баг."""
    torch.manual_seed(1)
    m = tiny(20)
    opt = m.configure_optimizer(3e-3)
    data = torch.randint(0, 20, (400,))
    x, y = get_batch(data, 4, 32, "cpu")
    first = m(x, y)[1].item()
    for _ in range(120):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_get_batch_target_is_shifted():
    data = torch.arange(100)
    x, y = get_batch(data, 5, 8, "cpu")
    assert torch.equal(y, x + 1)


def test_sample_lengths_and_greedy_is_deterministic():
    m = tiny().eval()
    idx = torch.zeros(1, 3, dtype=torch.long)
    out = sample(m, idx, 10, temperature=1.0, top_k=5)
    assert out.shape == (1, 13)
    g1 = sample(m, idx, 8, temperature=0)
    g2 = sample(m, idx, 8, temperature=0)
    assert torch.equal(g1, g2)
    # контекст длиннее block_size не ломает генерацию
    long_idx = torch.zeros(1, m.cfg.block_size + 5, dtype=torch.long)
    assert sample(m, long_idx, 2, temperature=0).shape[1] == m.cfg.block_size + 7


def test_top_p_runs():
    m = tiny().eval()
    out = sample(m, torch.zeros(1, 2, dtype=torch.long), 5, temperature=1.0, top_p=0.8)
    assert out.shape == (1, 7)


def test_lr_schedule():
    assert lr_at(0, 1000, 1e-3, 100) < lr_at(99, 1000, 1e-3, 100)     # разгон
    assert abs(lr_at(100, 1000, 1e-3, 100) - 1e-3) < 1e-9              # пик
    assert lr_at(1000, 1000, 1e-3, 100) < 0.2e-3                       # спад к минимуму


def test_config_roundtrip():
    c = GPTConfig(vocab_size=10, n_layer=1, n_head=1, n_embd=8)
    assert GPTConfig.from_dict(c.to_dict()) == c
