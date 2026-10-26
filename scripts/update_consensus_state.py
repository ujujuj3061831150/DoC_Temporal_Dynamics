# -*- coding: utf-8 -*-
for p in ['v3/src/temporal_dynamics/consensus.py', 'src/temporal_dynamics/consensus.py']:
    with open(p, 'r', encoding='utf-8') as f:
        t = f.read().replace('\r\n', '\n')
    
    old1 = '        return {\n            "type": "signed_standardized_mean",\n            "mean": self.scaler.mean_.tolist(),'
    new1 = '        return {\n            "type": "signed_standardized_mean",\n            "c": self.c,\n            "mean": self.scaler.mean_.tolist(),'
    assert old1 in t, f'old1 not in {p}'
    t = t.replace(old1, new1)
    
    old2 = '    def state(self) -> Dict[str, Any]:\n        return {\n            "type": "adaptive_gated_consensus",\n            "beta_base": self.beta_base,'
    new2 = '    def state(self) -> Dict[str, Any]:\n        return {\n            "type": "adaptive_gated_consensus",\n            "gating_type": self.gating_type,\n            "theta_0": self.theta_0,\n            "kappa": self.kappa,\n            "c": self.c,\n            "beta_base": self.beta_base,'
    assert old2 in t, f'old2 not in {p}'
    t = t.replace(old2, new2)
    
    with open(p, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(t)
    print(f'Successfully updated state serialization in {p}')
