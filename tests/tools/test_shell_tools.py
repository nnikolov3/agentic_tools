import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
from pathlib import Path
import logging
from src.tools.shell_tools import ShellTools

class TestShellTools(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.WARNING)
        self.config = {
            "readme_writer": {
                "prompt": "test_prompt",
                "model_name": "test_model",
                "temperature": 0.5,
                "description": "test_description",
                "model_provider": "test_provider",
                "alternative_model": "alt_model",
                "alternative_provider": "alt_provider",
                "skills": ["skill1", "skill2"],
            },
            "project_root": "/tmp/test_project",
            "design_docs": ["doc1.md", "doc2.md"],
            "source": "/tmp/test_source",
            "project_directories": ["/tmp/test_project/src"],
            "include_extensions": [".py", ".md"],
            "max_file_bytes": 1024,
            "exclude_directories": ["__pycache__"],
            "exclude_files": ["exclude_this.py"],
            "recent_minutes": 60,
        }
        self.agent = "readme_writer"
        self.shell_tools = ShellTools(self.agent, self.config)
        
        # Create dummy directories and files for testing
        self.test_project_dir = Path("/tmp/test_project/src")
        self.test_project_dir.mkdir(parents=True, exist_ok=True)
        (self.test_project_dir / "test_file.py").write_text("python test")
        (self.test_project_dir / "test_file.md").write_text("markdown test")
        (self.test_project_dir / "exclude_this.py").write_text("excluded")
        (self.test_project_dir / "__pycache__").mkdir(exist_ok=True)
        (self.test_project_dir / "__pycache__" / "cache.py").write_text("cached")


    def tearDown(self):
        # Clean up dummy directories and files
        if os.path.exists("/tmp/test_project"):
            shutil.rmtree("/tmp/test_project")

    def test_init(self):
        self.assertEqual(self.shell_tools.agent_config, self.config[self.agent])
        self.assertEqual(self.shell_tools.project_root, self.config["project_root"])

    def test_init_missing_agent(self):
        with self.assertRaises(ValueError):
            ShellTools("missing_agent", self.config)

    def test_concatenate_all_files(self):
        result = self.shell_tools.concatenate_all_files()
        expected = {
            'test_file.py': 'python test',
            'test_file.md': 'markdown test'
        }
        self.assertEqual(result, expected)

    def test_write_file(self):
        test_filepath = "/tmp/test_project/test_write.txt"
        test_content = "test content"
        
        # Test basic write
        self.shell_tools.write_file(test_filepath, test_content)
        with open(test_filepath, "r") as f:
            self.assertEqual(f.read(), test_content)

        # Test backup
        self.shell_tools.write_file(test_filepath, "new content", backup=True)
        backup_path = Path(test_filepath).with_suffix(".txt.bak")
        self.assertTrue(backup_path.exists())
        with open(backup_path, "r") as f:
            self.assertEqual(f.read(), test_content)

    @patch("subprocess.run")
    def test_get_git_info(self, mock_subprocess_run):
        # Mock git config user.name
        mock_subprocess_run.side_effect = [
            MagicMock(returncode=0, stdout="testuser"),
            MagicMock(returncode=0, stdout="https://github.com/test/test.git")
        ]
        
        git_info = self.shell_tools.get_git_info()
        self.assertEqual(git_info["username"], "testuser")
        self.assertEqual(git_info["url"], "https://github.com/test/test.git")

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_get_git_info_no_git(self, mock_subprocess_run):
        with self.assertRaises(FileNotFoundError):
            self.shell_tools.get_git_info()

    def test_cleanup_escapes(self):
        self.assertEqual(self.shell_tools.cleanup_escapes("a\\nb"), "a\nb")
        self.assertEqual(self.shell_tools.cleanup_escapes("a\nb"), "a\nb")
        self.assertEqual(self.shell_tools.cleanup_escapes("a\\tb"), "a\tb")
        self.assertEqual(self.shell_tools.cleanup_escapes("a\tb"), "a\tb")

    def test_cleanup_escapes_invalid(self):
        invalid_escape = "a\\x"
        with self.assertLogs('src.tools.shell_tools', level='WARNING') as cm:
            self.shell_tools.cleanup_escapes(invalid_escape)
            self.assertEqual(len(cm.output), 1)
            self.assertIn("Could not decode unicode escapes in string", cm.output[0])

if __name__ == '__main__':
    unittest.main()