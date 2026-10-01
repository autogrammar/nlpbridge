import copy
import unittest
from nlpbridge import select_operation, BridgeError, ModelError

URI = 'willman://operation/read/v1'
OPS = [{'uri': URI, 'desc': 'Read a document', 'effects': ['read'],
        'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                         'required': ['path'], 'additionalProperties': False}}]
READY = {'decision': {'status': 'ready', 'uri': URI, 'args': {'path': '文書.txt'}}}

class Model:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
    def generate(self, messages, schema):
        self.calls.append(copy.deepcopy((messages, schema)))
        value = self.answers.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

class SelectionTests(unittest.TestCase):
    def test_full_unicode_query_and_catalog_are_preserved(self):
        for text in ['Nie usuwaj dokumentu, odczytaj 文書.txt', '文書.txt を読んで', 'Lies 文書.txt, ohne sie zu löschen']:
            model = Model(READY)
            self.assertEqual(select_operation(text, OPS, model), READY['decision'])
            self.assertIn(text, model.calls[0][0][1]['content'])
    def test_wrong_uri_type_extra_keys_and_missing_argument_fail_closed(self):
        for fields in [{'uri': 'unknown'}, {'args': {'path': 42}}, {'args': {}}, {'shell': 'rm -rf .'}]:
            answer = {'decision': {**READY['decision'], **fields}}
            model = Model(answer, answer)
            with self.assertRaises(ModelError):
                select_operation('read', OPS, model)
            self.assertEqual(len(model.calls), 2)
    def test_one_repair(self):
        model = Model({'bad': True}, READY)
        self.assertEqual(select_operation('read', OPS, model)['status'], 'ready')
        self.assertEqual(len(model.calls), 2)
    def test_abstention_and_no_model(self):
        answer = {'decision': {'status': 'clarify', 'question': 'Which path?'}}
        self.assertEqual(select_operation('read', OPS, Model(answer)), answer['decision'])
        with self.assertRaises(ModelError):
            select_operation('read', OPS, None)
    def test_transport_never_retries_or_falls_back(self):
        model = Model(ModelError('offline'))
        with self.assertRaises(ModelError):
            select_operation('mail koru run', OPS, model)
        self.assertEqual(len(model.calls), 1)
    def test_catalog_and_prompt_limits(self):
        with self.assertRaises(BridgeError):
            select_operation('read', OPS * 2, Model())
        model = Model()
        with self.assertRaises(ModelError):
            select_operation('read', OPS, model, max_prompt_bytes=10)
        self.assertEqual(model.calls, [])
    def test_empty_catalog(self):
        self.assertEqual(select_operation('read', [], None)['status'], 'unsupported')
