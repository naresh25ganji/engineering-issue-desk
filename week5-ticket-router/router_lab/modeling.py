"""Local inference with the same prompt and decoder for every checkpoint.

Importing this module never imports torch, transformers or PEFT, and never loads
weights. ``LocalRouter.load`` (or the first prediction) performs those actions.
The optional sequence scorer sums conditional log probabilities for the entire
label and end marker. Its scores and greedy decisions are not calibrated
probabilities of correct routing.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import importlib
from pathlib import Path
import time
from typing import Any, Callable

from .config import LABELS, format_prompt, messages_for


END_MARKER = '<|im_end|>'
MAX_NEW_TOKENS = 16
CONFIDENCE_NOTE = 'No calibrated correctness probability is available.'
# This is the plain ChatML layout used by LLaMA Factory qwen3_nothink.
# Inference itself uses config.format_prompt, so a base tokenizer's bundled
# chat template cannot silently change the experiment.
CHAT_TEMPLATE = (
    r"{% for message in messages %}"
    r"{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    r"{% endfor %}"
    r"{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _letter_system_prompt() -> str:
    return ('You are an IT helpdesk ticket routing assistant. '
            'Classify the ticket by responding with ONLY a single letter — nothing else:\n'
            + '\n'.join(f'{letter}) {label}' for letter, label in zip('ABCDEFG', LABELS)))


def _prompt_for(text: str, decoding: str) -> str:
    messages = messages_for(text)
    if decoding == 'letter_choice':
        messages = [dict(message) for message in messages]
        messages[0]['content'] = _letter_system_prompt()
    return format_prompt(messages)


def _library(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f'Local model execution requires {name}; install the project training dependencies.'
        ) from exc


def _dtype(torch_module, value: str):
    names = {'float16': 'float16', 'fp16': 'float16', 'bfloat16': 'bfloat16',
             'bf16': 'bfloat16', 'float32': 'float32', 'fp32': 'float32'}
    if value == 'auto':
        return 'auto'
    if value not in names:
        raise ValueError('dtype must be auto, float16/fp16, bfloat16/bf16, or float32/fp32')
    return getattr(torch_module, names[value])


def _rows(value) -> list:
    return value.tolist() if hasattr(value, 'tolist') else value


def _device(model):
    device = getattr(model, 'device', None)
    if str(device) == 'meta' and hasattr(model, 'get_input_embeddings'):
        device = model.get_input_embeddings().weight.device
    return device


def _inputs(tokenizer, prompt: str, model) -> dict:
    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    device = _device(model)
    # Qwen3 has no token-type embedding. Generic saved fast tokenizers can emit
    # token_type_ids even though Qwen's generate() rejects that argument.
    return {key: value.to(device) if device is not None and hasattr(value, 'to') else value
            for key, value in encoded.items() if key in {'input_ids', 'attention_mask'}}


def _inference(torch_module):
    return torch_module.inference_mode() if hasattr(torch_module, 'inference_mode') else nullcontext()


def _synchronize(torch_module, model) -> None:
    if str(_device(model)).startswith('cuda'):
        torch_module.cuda.synchronize(_device(model))
    elif str(_device(model)).startswith('mps') and hasattr(torch_module, 'mps'):
        synchronize = getattr(torch_module.mps, 'synchronize', None)
        if synchronize is not None:
            synchronize()


def _configure(tokenizer, model) -> int:
    ids = tokenizer.encode(END_MARKER, add_special_tokens=False)
    if len(ids) != 1 or ids[0] == getattr(tokenizer, 'unk_token_id', None):
        raise ValueError('The router tokenizer must encode <|im_end|> as one known token.')
    eos_id = int(ids[0])
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.eos_token = END_MARKER
    if getattr(tokenizer, 'pad_token_id', None) is None:
        tokenizer.pad_token = END_MARKER
    stop_ids = _stop_ids(tokenizer, eos_id)
    for config in (getattr(model, 'config', None), getattr(model, 'generation_config', None)):
        if config is not None:
            config.eos_token_id = stop_ids
            pad_id = getattr(tokenizer, 'pad_token_id', None)
            config.pad_token_id = eos_id if pad_id is None else pad_id
    return eos_id


def _stop_ids(tokenizer, eos_id: int) -> list[int]:
    result = [eos_id]
    ids = tokenizer.encode('<|endoftext|>', add_special_tokens=False)
    if len(ids) == 1 and ids[0] != getattr(tokenizer, 'unk_token_id', None) and ids[0] not in result:
        result.append(int(ids[0]))
    return result


def _complete_label_stopping(tokenizer, prompt_length: int, torch_module):
    """Stop actual generation when its entire new text is an exact label."""
    transformers = _library('transformers')

    class CompleteLabelStopping(transformers.StoppingCriteria):
        def __init__(self):
            self.matched = False

        def __call__(self, input_ids, scores, **kwargs):
            matches = [tokenizer.decode(row[prompt_length:].tolist(), skip_special_tokens=False).strip() in LABELS
                       for row in input_ids]
            self.matched = self.matched or any(matches)
            return torch_module.tensor(matches, dtype=torch_module.bool, device=input_ids.device)

    criterion = CompleteLabelStopping()
    return criterion, transformers.StoppingCriteriaList([criterion])


class LocalRouter:
    """Load a base, local adapter, or merged model without changing its protocol.

    ``model``/``tokenizer``/``torch_module`` are dependency-injection seams for
    offline tests. When omitted, normal Hugging Face loading is used. Per-ticket
    latency includes tokenization and inference, and excludes initial loading.
    """

    def __init__(self, model_path: str, adapter_path: str | None = None,
                 revision: str | None = None, device: str = 'auto', dtype: str = 'auto',
                 decoding: str = 'generate', *, model=None, tokenizer=None,
                 torch_module=None):
        if decoding not in {'generate', 'label_stop', 'label_loglikelihood', 'letter_choice'}:
            raise ValueError('decoding must be generate, label_stop, label_loglikelihood, or letter_choice')
        if (model is None) != (tokenizer is None):
            raise ValueError('Inject model and tokenizer together.')
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.revision, self.device, self.dtype = revision, device, dtype
        self.decoding = decoding
        self._model, self._tokenizer, self._torch = model, tokenizer, torch_module
        self._ready = False
        self.load_seconds: float | None = None
        self._eos_id: int | None = None

    @property
    def prompt_metadata(self) -> dict[str, Any]:
        """Describe the actual prompt and decoder without loading a model."""
        import hashlib
        import json
        from .config import SYSTEM_PROMPT
        result = {'system_prompt': _letter_system_prompt() if self.decoding == 'letter_choice' else SYSTEM_PROMPT,
                  'template': 'qwen3_nothink', 'labels': list(LABELS),
                  'user_prefix': 'Support Ticket: ', 'decoding': self.decoding,
                  'eos_text': END_MARKER, 'max_new_tokens': MAX_NEW_TOKENS}
        if self.decoding == 'label_stop':
            result['stop_rule'] = {
                'type': 'complete_label_after_each_token_v1',
                'match': 'entire_generated_text.strip() exactly equals one allowed label',
                'requires_eos': False,
            }
        result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result

    def load(self) -> 'LocalRouter':
        if self._ready:
            return self
        started = time.perf_counter()
        self._torch = self._torch if self._torch is not None else _library('torch')
        if self._model is None:
            transformers = _library('transformers')
            kwargs = {'revision': self.revision, 'trust_remote_code': False}
            tokenizer_source = self.model_path
            tokenizer_kwargs = dict(kwargs)
            if self.adapter_path and (Path(self.adapter_path) / 'tokenizer_config.json').is_file():
                tokenizer_source = self.adapter_path
                tokenizer_kwargs.pop('revision')
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=_dtype(self._torch, self.dtype),
                device_map=self.device, **kwargs,
            )
            if self.adapter_path:
                self._model = _library('peft').PeftModel.from_pretrained(
                    self._model, self.adapter_path, is_trainable=False,
                )
        self._eos_id = _configure(self._tokenizer, self._model)
        self._model.eval()
        _synchronize(self._torch, self._model)
        self.load_seconds = time.perf_counter() - started
        self._ready = True
        return self

    def predict(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Ticket text must be a nonempty string.')
        self.load()
        _synchronize(self._torch, self._model)
        started = time.perf_counter()
        prompt = _prompt_for(text, self.decoding)
        inputs = _inputs(self._tokenizer, prompt, self._model)
        prompt_ids = _rows(inputs['input_ids'])[0]
        if not prompt_ids:
            raise ValueError('The router prompt encoded to no tokens.')
        if self.decoding in {'generate', 'label_stop'}:
            result = self._generate(inputs, len(prompt_ids))
        elif self.decoding == 'label_loglikelihood':
            result = self._score_labels(prompt, prompt_ids)
        else:
            result = self._letter_choice(inputs, len(prompt_ids))
        _synchronize(self._torch, self._model)
        result.update(decoding=self.decoding, latency_seconds=time.perf_counter() - started,
                      prompt_tokens=len(prompt_ids), confidence=None,
                      confidence_note=CONFIDENCE_NOTE)
        return result

    def _generate(self, inputs: dict, prompt_length: int) -> dict[str, Any]:
        pad_id = getattr(self._tokenizer, 'pad_token_id', None)
        criterion, extra = None, {}
        if self.decoding == 'label_stop':
            criterion, criteria = _complete_label_stopping(self._tokenizer, prompt_length, self._torch)
            extra['stopping_criteria'] = criteria
        with _inference(self._torch):
            sequences = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                eos_token_id=_stop_ids(self._tokenizer, self._eos_id),
                pad_token_id=self._eos_id if pad_id is None else pad_id,
                repetition_penalty=1.0, return_dict_in_generate=False,
                **extra,
            )
        generated_ids = _rows(sequences)[0][prompt_length:]
        raw = self._tokenizer.decode(generated_ids, skip_special_tokens=False)
        ended = bool(generated_ids) and generated_ids[-1] in _stop_ids(self._tokenizer, self._eos_id)
        content_ids = generated_ids[:-1] if ended else generated_ids
        parsed = self._tokenizer.decode(content_ids, skip_special_tokens=False).strip()
        complete_label = bool(criterion is not None and criterion.matched and parsed in LABELS and not ended)
        prediction = parsed if (ended or complete_label) and parsed in LABELS else None
        limit = not ended and not complete_label and len(generated_ids) >= MAX_NEW_TOKENS
        reason = None if prediction is not None else ('output_limit' if limit else 'invalid_label' if ended else 'missing_terminator')
        return {'prediction': prediction, 'valid': prediction is not None,
                'raw_output': raw, 'parsed_output': parsed, 'invalid_reason': reason,
                'finish_reason': 'eos' if ended else 'complete_label' if complete_label else 'length' if limit else 'other_stop',
                'token_limit_reached': limit,
                'input_tokens': prompt_length, 'output_tokens': len(generated_ids),
                'scored_tokens': 0, 'label_scores': None, 'score_type': None}

    def _letter_choice(self, inputs: dict, prompt_length: int) -> dict[str, Any]:
        choice_ids = []
        for letter in 'ABCDEFG':
            spaced = self._tokenizer.encode(' ' + letter, add_special_tokens=False)
            ids = spaced if len(spaced) == 1 else self._tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == getattr(self._tokenizer, 'unk_token_id', None):
                raise ValueError(f'Letter {letter} must encode to one known token.')
            choice_ids.append(ids[0])
        if len(set(choice_ids)) != len(LABELS):
            raise ValueError('Letter-choice token IDs must be distinct.')
        with _inference(self._torch):
            logits = self._model(**inputs).logits[0, -1, choice_ids].float().tolist()
        if not all(_finite(float(value)) for value in logits):
            raise ValueError('Nonfinite letter-choice logits; no prediction was accepted.')
        winner = max(range(len(LABELS)), key=lambda i: logits[i])
        return {'prediction': LABELS[winner], 'valid': True, 'raw_output': None,
                'parsed_output': 'ABCDEFG'[winner], 'invalid_reason': None,
                'finish_reason': 'candidate_scoring', 'token_limit_reached': False,
                'input_tokens': prompt_length, 'output_tokens': 0, 'scored_tokens': len(LABELS),
                'label_scores': dict(zip(LABELS, map(float, logits))),
                'score_type': 'constrained_letter_logits'}

    def _score_labels(self, prompt: str, prompt_ids: list[int]) -> dict[str, Any]:
        scores: dict[str, float] = {}
        processed_tokens = scored_tokens = 0
        for label in LABELS:
            inputs = _inputs(self._tokenizer, prompt + label + END_MARKER, self._model)
            full_ids = _rows(inputs['input_ids'])[0]
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError('Candidate tokenization changed the shared prompt prefix.')
            targets = inputs['input_ids'][0, len(prompt_ids):]
            with _inference(self._torch):
                outputs = self._model(**inputs)
                # Position k predicts token k+1. Include every label token and EOS.
                logits = outputs.logits[0, len(prompt_ids) - 1:-1, :].float()
                log_probs = self._torch.log_softmax(logits, dim=-1)
                selected = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                score = float(selected.sum().item())
            if not _finite(score):
                raise ValueError('Nonfinite candidate log likelihood; no prediction was accepted.')
            scores[label] = score
            processed_tokens += len(full_ids)
            scored_tokens += len(full_ids) - len(prompt_ids)
        prediction = max(scores, key=scores.get)
        return {'prediction': prediction, 'valid': True, 'raw_output': None,
                'parsed_output': prediction, 'invalid_reason': None,
                'finish_reason': 'candidate_scoring', 'token_limit_reached': False,
                'input_tokens': processed_tokens,
                'output_tokens': 0, 'scored_tokens': scored_tokens,
                'label_scores': scores, 'score_type': 'sequence_loglikelihood_sum'}


def _finite(value: float) -> bool:
    import math
    return math.isfinite(value)


def merge_adapter(base_model_path: str, adapter_path: str, output_dir: str | Path, *,
                  revision: str | None = None, device: str = 'cpu', dtype: str = 'auto',
                  model_loader: Callable | None = None, tokenizer_loader: Callable | None = None,
                  adapter_loader: Callable | None = None, torch_module=None) -> dict[str, Any]:
    """Safely merge, save and freshly reload a standalone checkpoint.

    A nonempty output directory is rejected. Verification checks tokenizer
    identity on the shared prompt and a finite forward pass from fresh weights;
    it does not establish routing accuracy or replace the held-out evaluation.
    """
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Merge output must be a new or empty directory.')
    torch_module = torch_module if torch_module is not None else _library('torch')
    if model_loader is None or tokenizer_loader is None:
        transformers = _library('transformers')
        model_loader = model_loader or transformers.AutoModelForCausalLM.from_pretrained
        tokenizer_loader = tokenizer_loader or transformers.AutoTokenizer.from_pretrained
    adapter_loader = adapter_loader or _library('peft').PeftModel.from_pretrained
    kwargs = {'revision': revision, 'trust_remote_code': False}
    tokenizer_source = adapter_path if (Path(adapter_path) / 'tokenizer_config.json').is_file() else base_model_path
    tokenizer_kwargs = kwargs if tokenizer_source == base_model_path else {'trust_remote_code': False}
    tokenizer = tokenizer_loader(tokenizer_source, **tokenizer_kwargs)
    model_kwargs = {'torch_dtype': _dtype(torch_module, dtype), 'device_map': device, **kwargs}
    base = model_loader(base_model_path, **model_kwargs)
    _configure(tokenizer, base)
    adapted = adapter_loader(base, adapter_path, is_trainable=False)
    merged = adapted.merge_and_unload(safe_merge=True)
    _configure(tokenizer, merged)
    probe = format_prompt(messages_for('Please install the requested application.'))
    expected_ids = tokenizer.encode(probe, add_special_tokens=False)
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(destination), safe_serialization=True)
    tokenizer.save_pretrained(str(destination))
    del merged, adapted, base, tokenizer
    gc.collect()
    if str(device).startswith('cuda') and hasattr(torch_module, 'cuda'):
        torch_module.cuda.empty_cache()
    # Read from the saved directory, rather than trusting the still-live merge.
    fresh_tokenizer = tokenizer_loader(str(destination), trust_remote_code=False, local_files_only=True)
    fresh = model_loader(str(destination), torch_dtype=_dtype(torch_module, dtype),
                         device_map=device, trust_remote_code=False, local_files_only=True)
    if fresh_tokenizer.chat_template != CHAT_TEMPLATE or fresh_tokenizer.eos_token != END_MARKER:
        raise ValueError('Saved tokenizer did not preserve the router chat template and end marker.')
    _configure(fresh_tokenizer, fresh)
    if fresh_tokenizer.encode(probe, add_special_tokens=False) != expected_ids:
        raise ValueError('Saved tokenizer changed the shared prompt token IDs.')
    fresh.eval()
    with _inference(torch_module):
        logits = fresh(**_inputs(fresh_tokenizer, probe, fresh)).logits
        finite = bool(torch_module.isfinite(logits).all().item())
    if not finite:
        raise ValueError('Freshly loaded merged weights produced nonfinite logits.')
    del fresh, fresh_tokenizer
    gc.collect()
    return {'output_dir': str(destination.resolve()), 'base_model': str(base_model_path),
            'base_revision': revision, 'adapter_path': str(adapter_path),
            'verified_reload': True, 'tokenizer_verified': True, 'finite_forward': True,
            'verification_note': 'Fresh local reload and finite forward pass; accuracy remains unmeasured.'}
