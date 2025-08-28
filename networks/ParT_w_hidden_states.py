import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

        # --- TCAV state (ADD THESE 2 LINES) ---
        self._tcav_handle = None
        self._tcav_acts = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)
    
    def get_loss(data_config, **kwargs):
        return torch.nn.CrossEntropyLoss()

    # ---------- TCAV helpers (no dataloader inside) ----------
    def _resolve(self, dotted: str):
        """Resolve 'mod.blocks.3.attn' style paths."""
        m = self
        for a in dotted.split('.'):
            m = getattr(m, a)
        return m

    def register_tcav_layer(self, dotted: str, capture: str = "input"):
        if self._tcav_handle is not None:
            self._tcav_handle.remove()
        m = self._resolve(dotted)

        def _hook(_, inp, out):
            # capture 'input' to LayerNorm instead of its output
            self._tcav_acts = inp[0] if capture == "input" else out

        self._tcav_handle = m.register_forward_hook(_hook)

    def remove_tcav_layer(self):
        if self._tcav_handle is not None:
            self._tcav_handle.remove()
        self._tcav_handle, self._tcav_acts = None, None

    def _forward_get_logits(self, points, features, lorentz_vectors, mask):
        """
        Returns logits regardless of base model’s usual output.
        Tries return_logits=True; else toggles for_inference=False.
        Also unwraps (output, hiddens) if your model returns a tuple.
        """
        # try kwargs path first (if your ParticleTransformer supports it)
        try:
            out = self.mod(features, v=lorentz_vectors, mask=mask, return_logits=True)
        except TypeError:
            # fallback: temporarily force logits via for_inference flag
            had_attr = hasattr(self.mod, 'for_inference')
            old_flag = getattr(self.mod, 'for_inference', False)
            if had_attr: self.mod.for_inference = False
            out = self.mod(features, v=lorentz_vectors, mask=mask)
            if had_attr: self.mod.for_inference = old_flag

        # unwrap tuple like (output, hiddens)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out  # logits

    @torch.no_grad()
    def get_acts_from_batch(self, points, features, lorentz_vectors, mask) -> torch.Tensor:
        """Run a forward to populate _tcav_acts. Returns flattened activations (B, D), detached."""
        assert self._tcav_handle is not None, "Call register_tcav_layer first."
        _ = self.forward(points, features, lorentz_vectors, mask)  # probs/logits ok; hook captures acts
        A = self._tcav_acts
        return A.detach().reshape(A.size(0), -1)

    def tcav_grads_from_batch(self, points, features, lorentz_vectors, mask, class_idx: int) -> torch.Tensor:
        """
        Compute per-sample ∂(logit_k)/∂a flattened for current batch.
        Returns (B, D) tensor on CPU (detached).
        """
        assert self._tcav_handle is not None, "Call register_tcav_layer first."
        self.zero_grad(set_to_none=True)
        logits = self._forward_get_logits(points, features, lorentz_vectors, mask)  # (B, C)
        target = logits[:, class_idx]
        # grads wrt hooked activations (shape (1, N, C)))
        grads = torch.autograd.grad(
            outputs=target,
            inputs=self._tcav_acts,                            # (1, N, C) — the hooked tensor
            grad_outputs=torch.ones_like(target),                # equivalent to .sum()
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]   
        # assume latent is channel first
        grads = grads.permute(1, 0, 2)
        print("grads", grads[:, 0])
        print("acts", self._tcav_acts.permute(1, 0, 2)[:, 0])
        return grads.reshape(grads.size(0), -1).detach().cpu()


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        hidden_states=True,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


