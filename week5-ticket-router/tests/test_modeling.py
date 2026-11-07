from __future__ import annotations

import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from router_lab.config import LABELS, format_prompt, messages_for
from router_lab.modeling import CHAT_TEMPLATE, END_MARKER, LocalRouter, merge_adapter


@pytest.fixture
def torch():
    module = pytest.importorskip('torch')
    original = module.get_num_threads()
    module.set_num_threads(1)
    yield module
    module.set_num_threads(original)


class CharTokenizer:
    """A deterministic offline tokenizer; special tokens remain indivisible."""
    def __init__(self, torch):
        self.torch = torch
        self.vocab = {'<pad>': 0, '<|im_start|>': 1, END_MARKER: 2, '<|endoftext|>': 3}
        self.pad_token_id, self.unk_token_id = 0, None
        self.eos_token = '<|endoftext|>'
        self.chat_template = None

    def encode(self, text, add_special_tokens=False):
        pieces = re.findall(r'<\|(?:im_start|im_end|endoftext)\|>|.', text, re.DOTALL)
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
        return [self.vocab[piece] for piece in pieces]

    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.vocab.items()}
        return ''.join(inverse[int(value)] for value in ids)

    def __call__(self, text, return_tensors, add_special_tokens):
        ids = self.torch.tensor([self.encode(text)], dtype=self.torch.long)
        return {'input_ids': ids, 'attention_mask': self.torch.ones_like(ids)}


class ScriptedModel:
    def __init__(self, torch, tokenizer, generated='Software', terminator=END_MARKER):
        self.torch, self.tokenizer = torch, tokenizer
        self.generated, self.terminator = generated, terminator
        self.device = 'cpu'
        self.config, self.generation_config = SimpleNamespace(), SimpleNamespace()
        self.calls, self.generation_calls = [], []

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generation_calls.append(kwargs)
        prompt_ids = kwargs['input_ids'][0].tolist()
        suffix = self.tokenizer.encode(self.generated + self.terminator)
        return self.torch.tensor([prompt_ids + suffix])

    def __call__(self, input_ids, **kwargs):
        ids = input_ids[0].tolist()
        self.calls.append(ids)
        text = self.tokenizer.decode(ids)
        suffix = text.rsplit('<|im_start|>assistant\n', 1)[-1]
        logits = self.torch.zeros(1, len(ids), 512)
        if suffix:
            # A real teacher-forcing shape: position k predicts token k+1.
            strength = 14.0 if suffix == 'Software' + END_MARKER else 0.0
            for position, next_id in enumerate(ids[1:]):
                logits[0, position, next_id] = strength
        else:
            logits[0, -1, self.tokenizer.encode('F')[0]] = 9.0
        return SimpleNamespace(logits=logits)


def router(torch, *, output='Software', terminator=END_MARKER, decoding='generate'):
    tokenizer = CharTokenizer(torch)
    model = ScriptedModel(torch, tokenizer, output, terminator)
    return LocalRouter('unused', model=model, tokenizer=tokenizer, torch_module=torch,
                       decoding=decoding), model, tokenizer


def test_module_import_does_not_load_inference_libraries():
    result = subprocess.run(
        [sys.executable, '-c',
         "import sys; import router_lab.modeling; "
         "assert not {'torch', 'transformers', 'peft'}.intersection(sys.modules)"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_greedy_uses_shared_prompt_and_keeps_raw_output(torch):
    classifier, model, tokenizer = router(torch)
    result = classifier.predict('Install Acrobat.')
    assert result['prediction'] == 'Software' and result['valid']
    assert result['raw_output'] == 'Software' + END_MARKER
    assert result['parsed_output'] == 'Software'
    assert result['confidence'] is None and 'calibrated' in result['confidence_note']
    assert result['input_tokens'] == result['prompt_tokens']
    assert result['output_tokens'] == len(tokenizer.encode('Software' + END_MARKER))
    request = model.generation_calls[0]
    assert tokenizer.decode(request['input_ids'][0].tolist()) == format_prompt(messages_for('Install Acrobat.'))
    assert request['eos_token_id'] == [2, 3]
    assert request['do_sample'] is False and request['num_beams'] == 1
    assert 'stopping_criteria' not in request
    assert model.config.pad_token_id == 0
    assert result['latency_seconds'] >= 0 and not result['token_limit_reached']


@pytest.mark.parametrize('output', ['software', 'Software.','Software\nBecause this is an install.','Unknown'])
def test_nonexact_generation_is_invalid_without_general_fallback(torch, output):
    classifier, _, _ = router(torch, output=output)
    result = classifier.predict('A ticket')
    assert result['prediction'] is None and result['valid'] is False
    assert result['invalid_reason'] == 'invalid_label'
    assert result['raw_output'] == output + END_MARKER


def test_endoftext_is_a_valid_stop_and_truncated_text_stays_invalid(torch):
    classifier, _, _ = router(torch, terminator='<|endoftext|>')
    assert classifier.predict('A ticket')['prediction'] == 'Software'
    classifier, _, _ = router(torch, output='x' * 16, terminator='')
    result = classifier.predict('A ticket')
    assert result['prediction'] is None
    assert result['finish_reason'] == 'length' and result['token_limit_reached']
    assert result['invalid_reason'] == 'output_limit'


def test_label_stop_has_explicit_protocol_without_changing_original_generation_metadata():
    original = LocalRouter('unused').prompt_metadata
    assert original['sha256'] == '52c4748ddb48ebe9bfd8c95d645fdca000bbc36648d8752bac9b7e1e5a0d16fb'
    assert 'stop_rule' not in original
    repaired = LocalRouter('unused', decoding='label_stop').prompt_metadata
    assert repaired['sha256'] != original['sha256']
    assert repaired['decoding'] == 'label_stop'
    assert repaired['stop_rule']['type'] == 'complete_label_after_each_token_v1'
    assert repaired['stop_rule']['requires_eos'] is False


@pytest.mark.parametrize('suffix, expected, finish', [
    ('Softwarejunk' + END_MARKER, 'Software', 'complete_label'),
    (' Softwarejunk' + END_MARKER, 'Software', 'complete_label'),
    ('not Software' + END_MARKER, None, 'eos'),
    ('Soft' + END_MARKER, None, 'eos'),
])
def test_label_stop_runs_inside_actual_hf_generation(torch, suffix, expected, finish):
    transformers = pytest.importorskip('transformers')
    tokenizer = CharTokenizer(torch)
    forced_ids = tokenizer.encode(suffix)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=512, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=2048, pad_token_id=0, eos_token_id=2,
        tie_word_embeddings=True, attention_dropout=0.0,
    )).eval()
    original_generate = model.generate
    steps = []

    class ForceTokens(transformers.LogitsProcessor):
        def __init__(self, prompt_length):
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores):
            position = input_ids.shape[-1] - self.prompt_length
            next_id = forced_ids[position] if position < len(forced_ids) else 2
            steps.append(next_id)
            scores.fill_(float('-inf'))
            scores[:, next_id] = 0
            return scores

    def generate(**kwargs):
        return original_generate(**kwargs, logits_processor=[ForceTokens(kwargs['input_ids'].shape[-1])])

    model.generate = generate
    classifier = LocalRouter('unused', model=model, tokenizer=tokenizer,
                             torch_module=torch, decoding='label_stop')
    result = classifier.predict('A ticket')
    assert result['prediction'] == expected
    assert result['finish_reason'] == finish
    assert result['output_tokens'] == len(steps)
    if expected:
        assert result['raw_output'].strip() == expected
        assert result['valid'] and not result['token_limit_reached']
        assert END_MARKER not in result['raw_output'] and 'junk' not in result['raw_output']
        assert len(steps) < len(forced_ids)  # The unwanted continuation was never generated.
        steps.clear()
        classifier.decoding = 'generate'
        original = classifier.predict('A ticket')
        assert original['prediction'] is None
        assert original['raw_output'] == suffix
        assert original['finish_reason'] == 'eos'
    else:
        assert not result['valid'] and result['invalid_reason'] == 'invalid_label'
        assert result['raw_output'] == suffix


def test_label_stop_does_not_accept_unterminated_text_if_generator_ignores_rule(torch):
    classifier, _, _ = router(torch, output='Software', terminator='', decoding='label_stop')
    result = classifier.predict('A ticket')
    assert result['prediction'] is None and result['invalid_reason'] == 'missing_terminator'
    assert result['finish_reason'] == 'other_stop'


def test_full_candidate_likelihood_uses_all_label_tokens_and_eos(torch):
    classifier, model, tokenizer = router(torch, decoding='label_loglikelihood')
    result = classifier.predict('A ticket')
    assert result['prediction'] == 'Software'
    assert result['score_type'] == 'sequence_loglikelihood_sum'
    assert result['raw_output'] is None and result['output_tokens'] == 0
    assert tuple(result['label_scores']) == LABELS
    assert len(model.calls) == len(LABELS)
    prompt = format_prompt(messages_for('A ticket'))
    for ids, label in zip(model.calls, LABELS):
        assert tokenizer.decode(ids) == prompt + label + END_MARKER
    assert result['input_tokens'] == sum(map(len, model.calls))
    assert result['scored_tokens'] == sum(len(tokenizer.encode(label + END_MARKER)) for label in LABELS)
    # Off-by-one scoring cannot match this: the high logits predict the NEXT token.
    assert result['label_scores']['Software'] > -0.02


def test_letter_choice_is_explicitly_a_different_prompt_protocol(torch):
    classifier, _, _ = router(torch, decoding='letter_choice')
    result = classifier.predict('A ticket')
    assert result['prediction'] == 'Active Directory' and result['parsed_output'] == 'F'
    assert result['score_type'] == 'constrained_letter_logits'
    assert result['confidence'] is None and result['output_tokens'] == 0
    normal, _, _ = router(torch)
    assert classifier.prompt_metadata['sha256'] != normal.prompt_metadata['sha256']
    assert 'F) Active Directory' in classifier.prompt_metadata['system_prompt']


def test_nonfinite_candidate_is_rejected(torch):
    classifier, model, _ = router(torch, decoding='label_loglikelihood')
    original_call = model.__class__.__call__

    class NonfiniteModel(ScriptedModel):
        def __call__(self, *args, **kwargs):
            result = original_call(self, *args, **kwargs)
            result.logits[:] = float('nan')
            return result

    model.__class__ = NonfiniteModel
    with pytest.raises(ValueError, match='Nonfinite'):
        classifier.predict('A ticket')


def test_configuration_and_merge_destination_fail_before_loading(tmp_path):
    with pytest.raises(ValueError, match='decoding'):
        LocalRouter('unused', decoding='unknown')
    with pytest.raises(ValueError, match='together'):
        LocalRouter('unused', model=object())
    destination = tmp_path / 'occupied'
    destination.mkdir()
    (destination / 'keep.txt').write_text('Keep this file.')
    with pytest.raises(ValueError, match='new or empty'):
        merge_adapter('unused', 'unused', destination)
    assert (destination / 'keep.txt').read_text() == 'Keep this file.'


def test_tiny_random_qwen3_adapter_merge_and_fresh_reload(torch, tmp_path):
    transformers = pytest.importorskip('transformers')
    peft = pytest.importorskip('peft')
    tokenizers = pytest.importorskip('tokenizers')
    vocab = {'<pad>': 0, '<unk>': 1, '<|im_start|>': 2, END_MARKER: 3, '<|endoftext|>': 4}
    vocab.update({chr(i): len(vocab) + j for j, i in enumerate(range(32, 127))})
    vocab['\n'] = len(vocab)
    tokenizer_backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer_backend.pre_tokenizer = tokenizers.pre_tokenizers.Split(tokenizers.Regex(''), behavior='isolated')
    tokenizer_backend.decoder = tokenizers.decoders.Fuse()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend, pad_token='<pad>', unk_token='<unk>',
        eos_token='<|endoftext|>', additional_special_tokens=['<|im_start|>', END_MARKER],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    assert tokenizer.apply_chat_template(messages_for('A ticket'), tokenize=False, add_generation_prompt=True) == format_prompt(messages_for('A ticket'))
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=2048, pad_token_id=0,
        eos_token_id=tokenizer.encode(END_MARKER, add_special_tokens=False)[0],
        tie_word_embeddings=True, attention_dropout=0.0,
    )
    torch.manual_seed(42)
    model = transformers.Qwen3ForCausalLM(config).eval()
    base = tmp_path / 'base'
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    model.config._name_or_path = str(base)
    model.name_or_path = str(base)
    adapted = peft.get_peft_model(model, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
        task_type='CAUSAL_LM', lora_dropout=0.0,
    )).eval()
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(0.02)
    adapter = tmp_path / 'adapter'
    adapted.save_pretrained(adapter)
    # A tokenizer saved by training should be the one carried into the merge.
    tokenizer.save_pretrained(adapter)
    probe = tokenizer(format_prompt(messages_for('A ticket')), return_tensors='pt', add_special_tokens=False)
    with torch.no_grad():
        expected = adapted(**probe).logits.detach().clone()
    adapter_router = LocalRouter(str(base), adapter_path=str(adapter), device='cpu',
                                 dtype='float32', decoding='label_loglikelihood')
    adapter_result = adapter_router.predict('A ticket')
    output = tmp_path / 'merged'
    report = merge_adapter(str(base), str(adapter), output, device='cpu', dtype='float32')
    assert report['verified_reload'] and report['finite_forward'] and report['tokenizer_verified']
    assert not (output / 'adapter_config.json').exists()
    fresh = transformers.AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
    with torch.no_grad():
        actual = fresh(**probe).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    # Exercise both actual inference implementations, using only random local weights.
    scorer = LocalRouter(str(output), device='cpu', dtype='float32', decoding='label_loglikelihood')
    scored = scorer.predict('A ticket')
    assert scored['prediction'] in LABELS
    assert scored['label_scores'] == pytest.approx(adapter_result['label_scores'], rel=1e-5, abs=1e-5)
    generator = LocalRouter(str(output), device='cpu', dtype='float32')
    generated = generator.predict('A ticket')
    assert generated['prediction'] is None or generated['prediction'] in LABELS
    assert isinstance(generated['raw_output'], str)
