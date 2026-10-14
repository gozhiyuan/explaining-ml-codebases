"""Regressions for doc validation without prescribed diagram types."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('doc_checker', ROOT / 'scripts/check_explain_doc.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DocSetChecks(unittest.TestCase):
    def test_deep_docs_do_not_require_arbitrary_diagram_types(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / 'index.md'
            detail = Path(temp) / 'requests.md'
            index.write_text('# Request service\n\nA component table explains the responsibilities.\n')
            detail.write_text('# Request path\n\n1. Receive input.\n2. Produce a result.\n')
            report = checker.Report()
            checker.check_doc_set([index, detail], 'deep', report)
            self.assertEqual(report.items, [])

    def test_shared_shape_reference_remains_required(self):
        with tempfile.TemporaryDirectory() as temp:
            index = Path(temp) / 'index.md'
            detail = Path(temp) / 'model.md'
            index.write_text('# Model docs\n')
            detail.write_text('# Tensor example\n\nHidden vectors have shape [B,T,D].\n')
            report = checker.Report()
            checker.check_doc_set([index, detail], 'deep', report)
            self.assertEqual(len(report.items), 1)
            self.assertEqual(report.items[0]['level'], 'WARN')
            self.assertEqual(report.items[0]['doc'], str(index))
            index.write_text('# Model docs\n\n## Shapes cheat sheet\n\nB=batch, T=positions, D=channels.\n')
            report = checker.Report()
            checker.check_doc_set([index, detail], 'deep', report)
            self.assertEqual(report.items, [])


if __name__ == '__main__':
    unittest.main()
