"""Narrow guard for credential-value requests in missing_information only.

This is not a general secret detector or semantic reviewer. It recognizes
explicit disclosure requests and credential-bearing return-value requests,
including the observed _configuration() (key, base) case. It neither examines
nor rewrites factual findings, source text, or citation quotes.
"""
from __future__ import annotations

import re


POLICY_VERSION = 1
SAFE_CONFIGURATION_REQUEST = (
    'Confirm whether the required credential is configured (yes/no); provide the provider name, '
    'configuration variable names, an endpoint without credentials or secret query parameters, '
    'and redacted error messages. Do not share credential values or other secrets.'
)
SOURCE_REDACTION_NOTE = 'Redact credential values and other secrets from any supplied source.'
_CREDENTIAL = (
    r'(?:api[ _-]?keys?|access[ _-]?tokens?|auth(?:entication|orization)?[ _-]?tokens?|'
    r'bearer[ _-]?tokens?|client[ _-]?secrets?|private[ _-]?keys?|passwords?|credentials?|secrets?|'
    r'[a-z][a-z0-9_]*(?:_api_key|_access_token|_auth_token|_secret|_password|_private_key))'
)
_CREDENTIAL_REFERENCE = re.compile(r'\b' + _CREDENTIAL + r'\b', re.I)
_DISCLOSURE_START = re.compile(
    r'^\s*(?:(?:please|kindly)\s+)?(?:provide|share|paste|send|show|reveal|print|dump|supply|'
    r'disclose|include|request|need|missing|required)\b|'
    r'^\s*(?:can|could|would)\s+you\s+(?:provide|share|paste|send|show|reveal|print|dump|supply|disclose)\b', re.I)
_VALUE_QUESTION_START = re.compile(r'^\s*(?:what|which)\s+(?:is|are)\b', re.I)
_OUTPUT_START = re.compile(
    r'^\s*(?:the\s+)?(?:(?:actual|specific|full|raw|exact|current|runtime)\s+)*(?:values?|contents?|outputs?|results?)\b', re.I)
_SHORT_CREDENTIAL_REQUEST = re.compile(
    r'^\s*(?:(?:the|your|actual|full|raw|exact|current)\s+)*' + _CREDENTIAL
    + r'(?:\s+(?:values?|contents?))?\s*[?.]?\s*$', re.I)
_RAW_VALUES = re.compile(
    r'\b(?:actual|specific|raw|full|exact|unredacted|literal)\s+(?:[a-z_]+\s+){0,3}'
    r'(?:values?|contents?|outputs?|credentials?|api[ _-]?keys?|tokens?|passwords?|secrets?)\b', re.I)
_VALUE_WORDS = re.compile(r'\b(?:values?|contents?|outputs?|results?)\b', re.I)
_EXPLICIT_CREDENTIAL_VALUE = re.compile(
    r'\b' + _CREDENTIAL + r'\s+(?:values?|contents?)\b|'
    r'\b(?:values?|contents?)\s+(?:of|for)\s+(?:(?:the|your|actual)\s+)*' + _CREDENTIAL + r'\b', re.I)
_PRESENCE = re.compile(
    r'\b(?:presence|yes/no|yes or no)\b|\b(?:whether|if)\b.{0,120}\b(?:configured|present|set|available|exists?)\b|'
    r'\b(?:is|are)\b.{0,80}\bconfigured\b', re.I)
_NAMES = re.compile(r'\b(?:variable names?|key names?|credential names?|names? of (?:the )?(?:credentials?|secrets?|api[ _-]?keys?))\b', re.I)
_HELPER_SOURCE = re.compile(
    r'\b(?:source code|implementation|function body)\b|'
    r'\b(?:source|definition)\s+(?:for|of)\s+[`\'"]?[a-z_]\w*\s*\(', re.I)
_CONFIG_HELPER = re.compile(r'(?<!\w)_configuration\s*\(\s*\)', re.I)
_KEY_HELPER = re.compile(r'(?<!\w)(?:_runtime_key|get_api_key|get_access_token)\s*\(\s*\)', re.I)
_KEY_RETURN = re.compile(r'\bkey\b', re.I)
_EXISTING_REDACTION = re.compile(
    r'\b(?:redact(?:ed|ing)? (?:all )?(?:credential values|credentials|secrets)|'
    r'(?:credential values|credentials|secrets) (?:are |must be |should be )?redacted)\b', re.I)


def _sanitize_request(text: str) -> str:
    # The exact application-added suffix is guidance, not a second request for
    # the credential values it tells the user to redact. Analyze the original
    # request body again, so adding the suffix cannot exempt an unsafe request.
    body = text.removesuffix(' ' + SOURCE_REDACTION_NOTE)
    credential = bool(_CREDENTIAL_REFERENCE.search(body))
    output_request = bool(_OUTPUT_START.search(body))
    disclosure_request = bool(_DISCLOSURE_START.search(body))
    request_shape = (output_request or disclosure_request or bool(_SHORT_CREDENTIAL_REQUEST.fullmatch(body))
                     or bool(_VALUE_QUESTION_START.search(body) and _VALUE_WORDS.search(body)))
    raw_values = bool(_RAW_VALUES.search(body))
    explicit_values = bool(_EXPLICIT_CREDENTIAL_VALUE.search(body))
    helper_source = bool(_HELPER_SOURCE.search(body))
    presence_or_names = bool(_PRESENCE.search(body) or _NAMES.search(body))
    credential_output = bool(_VALUE_WORDS.search(body)) and bool(
        _KEY_HELPER.search(body) or (_CONFIG_HELPER.search(body) and _KEY_RETURN.search(body)))
    # Source inspection can describe a credential-bearing return statement
    # without requesting its runtime value. Preserve that inspection, with a
    # redaction instruction. Explicit raw-value disclosure still takes priority.
    unsafe = (request_shape and (credential or credential_output)
              and (not helper_source or raw_values or explicit_values)
              and (not presence_or_names or raw_values or explicit_values))
    if unsafe:
        return SAFE_CONFIGURATION_REQUEST
    if helper_source and not _EXISTING_REDACTION.search(text):
        guarded = f'{text.rstrip()} {SOURCE_REDACTION_NOTE}'
        return guarded if len(guarded) <= 1000 else (
            'Provide the requested helper source or implementation with all credential values and other secrets redacted.')
    return text


def sanitize_missing_information(report: dict) -> tuple[dict, int]:
    """Copy a validated report and return the number of changed request items.

    Inputs must already satisfy the report contract (at most eight plain-text
    requests). Counts contain no original text or credential values. Applying
    the guard twice is idempotent.
    """
    requests = report['missing_information']
    if (not isinstance(requests, list) or len(requests) > 8
            or any(not isinstance(item, str) for item in requests)):
        raise ValueError('Information requests must be a bounded text list')
    safe = [_sanitize_request(item) for item in requests]
    return {**report, 'missing_information': safe}, sum(before != after for before, after in zip(requests, safe, strict=True))
