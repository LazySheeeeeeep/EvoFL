"""Credential-safe model and native-tools connectivity probe (WSL)."""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evolutefl.llm.client import OpenAICompatibleClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-v4.1-flash')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    key = os.getenv('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek credential (not saved): ')
    result = {'requested_model': args.model, 'base_url': 'https://api.deepseek.com', 'time': time.time()}
    request = urllib.request.Request('https://api.deepseek.com/models', headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result['available_models'] = [entry.get('id') for entry in json.load(response).get('data', [])]
    except urllib.error.HTTPError as error:
        result['models_endpoint_status'] = error.code
    client = OpenAICompatibleClient(base_url='https://api.deepseek.com', api_key=key,
        model=args.model, temperature=0, max_tokens=512, timeout=90, max_attempts=1)
    tool = {'type': 'function', 'function': {'name': 'ready', 'description': 'Confirm API connectivity.',
            'parameters': {'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']}}}
    for choice in ('required_named', 'single_tool_auto'):
        start = time.monotonic()
        try:
            response = client.chat(messages=[{'role': 'user', 'content': 'Call ready with ok=true. No other output.'}],
                tools=[tool], tool_choice={'type': 'function', 'function': {'name': 'ready'}} if choice == 'required_named' else None)
            calls = response.get('tool_calls') or []
            valid = bool(calls and calls[0].get('function', {}).get('name') == 'ready'
                         and json.loads(calls[0]['function']['arguments']).get('ok') is True)
            result[choice] = {'passed': valid, 'seconds': round(time.monotonic() - start, 2),
                              'response_model': response.get('raw', {}).get('model'),
                              'usage': response.get('raw', {}).get('usage')}
        except Exception as error:
            # Error bodies can contain sensitive transport context; persist type/status only.
            result[choice] = {'passed': False, 'error_type': type(error).__name__,
                              'status_code': getattr(error, 'status_code', None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    key = None


if __name__ == '__main__':
    main()
