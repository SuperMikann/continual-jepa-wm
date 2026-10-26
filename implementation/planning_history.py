"""Confirmed policy-history and rollout implementation, applied only by install.py.

The source transforms accept the audited legacy layout or this exact repair;
they reject other layouts. Experiment runners never install these repairs.
"""

def prepare_planning_history(self, info, needs_flush=None):
    """Stack real block-boundary pixels and normalized actions already executed between them."""
    history_len = getattr(self.cfg, 'history_len', 1)
    if history_len <= 1:
        return info
    import numpy as np
    import torch
    from collections import deque

    n, block = self.env.num_envs, self.cfg.action_block
    if info['pixels'].shape[1] != 1 or info['action'].shape[1] != 1:
        raise ValueError('History accumulation expects one current frame and last executed action per env step')
    if getattr(self, '_planning_history', None) is None:
        self._planning_history = [deque(maxlen=(history_len-1)*block+1) for _ in range(n)]
    flush = np.zeros(n, dtype=bool) if needs_flush is None else np.asarray(needs_flush, dtype=bool).reshape(n)
    dead = np.asarray(info.get('terminated', np.zeros(n, dtype=bool)), dtype=bool).reshape(n)
    for i, buf in enumerate(self._planning_history):
        if flush[i]:
            buf.clear()
        buf.append((info['pixels'][i, -1].detach().clone(), info['action'][i, -1].detach().flatten().clone()))
    replan = [i for i in range(n) if not dead[i] and (flush[i] or len(self._action_buffer[i]) == 0)]
    if not replan:
        return info
    frames = min(history_len, max((len(self._planning_history[i])-1)//block+1 for i in replan))
    pixels, actions = [], []
    for i, buf in enumerate(self._planning_history):
        k = min(frames, (len(buf)-1)//block+1)
        indices = [len(buf)-1-off*block for off in reversed(range(k))]
        real = [buf[j][0] for j in indices]
        pixels.append(torch.stack([real[0]]*(frames-k)+real))
        if frames > 1:
            chunks = [torch.cat([buf[j][1] for j in range(end-block+1, end+1)]) for end in indices[1:]]
            zero = buf[-1][1].new_zeros(block*buf[-1][1].numel())
            chunks = [zero]*(frames-k)+chunks
            past = torch.stack(chunks)
            if i in replan and not torch.isfinite(past).all():
                raise ValueError('Executed action history contains non-finite values')
            actions.append(past)
    info['pixels'] = torch.stack(pixels)
    if frames > 1:
        info['action_history'] = torch.stack(actions)
    else:
        info.pop('action_history', None)
    return info


def policy_history_source(text):
    """Add history preparation to the installed legacy policy without replacing its action selection."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorldModelPolicy']
    if len(classes) != 1:
        raise ValueError('Expected one installed WorldModelPolicy')
    cls = classes[0]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    fixed = inspect.getsource(prepare_planning_history).replace('def prepare_planning_history(', 'def _prepare_planning_history(', 1)
    if '_prepare_planning_history' in methods:
        current = ast.get_source_segment(text, methods['_prepare_planning_history'])
        if ast.dump(ast.parse(textwrap.dedent(current))) != ast.dump(ast.parse(fixed)):
            raise ValueError('Existing policy history helper is a different version')
        if 'self._prepare_planning_history(info_dict, _planning_history_flush)' not in text:
            raise ValueError('Existing policy helper is not connected')
        return text
    get, setup = methods['get_action'], methods['set_env']
    body = ast.get_source_segment(text, get)
    if 'action_history' in body or '_history_buffer' in body or '_planning_history' in body:
        raise ValueError('Installed policy already contains a different history mechanism; no overwrite')
    assignments = [node for node in ast.walk(get) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'info_dict'
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == '_prepare_info']
    if len(assignments) != 1:
        raise ValueError('Unexpected policy preprocessing interface')
    def entry_line(node):
        first = node.body[0]
        return first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str) else node.lineno
    edits = [
        (entry_line(get), "        _planning_history_flush = info_dict.get('_needs_flush')\n"),
        (assignments[0].end_lineno, '        info_dict = self._prepare_planning_history(info_dict, _planning_history_flush)\n'),
        (entry_line(setup), '        self._planning_history = None\n'),
        (cls.end_lineno, '\n'+textwrap.indent(fixed.rstrip()+'\n', '    ')),
    ]
    lines = text.splitlines(keepends=True)
    for line, addition in sorted(edits, reverse=True):
        lines.insert(line, addition)
    updated = ''.join(lines)
    compile(updated, '<patched policy history>', 'exec')
    return updated


def future_action_rollout(self, info, action_sequence, history_size: int = None):
    """Roll out every strictly-future candidate block; past blocks come from action_history."""
    import torch
    from einops import rearrange

    if history_size is None:
        history_size = getattr(self.predictor, 'num_frames', 3)
    H = info['pixels'].size(2)
    B, S, T, A = action_sequence.shape
    past = info.get('action_history')
    if past is None:
        past = action_sequence.new_zeros(B, S, 0, A)
    if past.shape != (B, S, H-1, A):
        raise ValueError(f'Expected {H-1} executed action blocks between {H} context frames; got {tuple(past.shape)}')
    info['action'] = torch.cat([past, action_sequence[:, :, :1]], dim=2)
    if 'emb' not in info:
        initial = self.encode({k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)})
        info['emb'] = initial['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
    states = list(rearrange(info['emb'], 'b s ... -> (b s) ...').unbind(1))
    actions = self.action_encoder(torch.cat([
        rearrange(past, 'b s ... -> (b s) ...'),
        rearrange(action_sequence, 'b s ... -> (b s) ...')], dim=1))
    for t in range(T):
        lo = max(0, H+t-history_size)
        prediction = self.predict(torch.stack(states[lo:], dim=1), actions[:, lo:H+t])[:, -1]
        states.append(prediction)
    info['predicted_emb'] = rearrange(torch.stack(states, dim=1), '(b s) t d -> b s t d', b=B, s=S)
    return info


def future_rollout_source(text):
    """Replace only the LeWM rollout method, preserving all other installed source."""
    import ast
    import inspect
    import textwrap

    classes = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'LeWM']
    if len(classes) != 1:
        raise ValueError('Expected one installed LeWM class')
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'rollout']
    if len(methods) != 1 or methods[0].decorator_list:
        raise ValueError('Unexpected installed rollout definition')
    node = methods[0]
    installed = ast.get_source_segment(text, node)
    fixed = inspect.getsource(future_action_rollout).replace('def future_action_rollout(', 'def rollout(', 1)
    already_fixed = ast.dump(ast.parse(textwrap.dedent(installed))) == ast.dump(ast.parse(fixed))
    if not already_fixed and not all(s in installed for s in (
            'torch.split(action_sequence, [H, T - H]', 'n_steps = T - H', 'range(n_steps + 1)')):
        raise ValueError('Installed rollout differs from the confirmed legacy method; no automatic overwrite')
    lines = text.splitlines(keepends=True)
    replacement = textwrap.indent(fixed.rstrip()+'\n', '    ')
    updated = ''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])
    compile(updated, '<patched LeWM>', 'exec')
    return updated
