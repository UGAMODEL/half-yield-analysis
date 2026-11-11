"""Check installed command routing without starting a server or loading a model."""
import unittest
from unittest.mock import patch

from halfyield import cli


class CommandTests(unittest.TestCase):
    def test_no_arguments_open_browser_launcher(self):
        with patch('sys.argv', ['halfyield']), patch('halfyield.desktop.main') as launch:
            cli.main()
        launch.assert_called_once_with()

    def test_arguments_go_to_inference_parser(self):
        with patch('sys.argv', ['halfyield', '--help']), patch('halfyield.main.main') as run:
            cli.main()
        run.assert_called_once_with()
