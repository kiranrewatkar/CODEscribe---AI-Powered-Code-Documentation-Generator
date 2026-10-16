"""
tests/test_training_config.py

Unit test suite for training configuration, parameter validation, and hardware detection.
"""

from pathlib import Path
import tempfile
import unittest

from training.training_config import TrainingExperimentConfig, detect_hardware


class TestTrainingConfig(unittest.TestCase):
    """Test suite for TrainingExperimentConfig and hardware detection."""

    def test_default_configuration_values(self):
        config = TrainingExperimentConfig()
        self.assertEqual(config.model_name_or_path, "Salesforce/codet5-small")
        self.assertEqual(config.tokenizer_name_or_path, "Salesforce/codet5-small")
        self.assertEqual(config.task_prefix, "document python code: ")
        self.assertEqual(config.max_source_length, 512)
        self.assertEqual(config.max_target_length, 128)
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.learning_rate, 5e-5)
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertEqual(config.seed, 42)

    def test_configuration_serialization_and_deserialization(self):
        config = TrainingExperimentConfig(
            num_train_epochs=5,
            learning_rate=3e-5,
            seed=123,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_config.json"
            config.save(save_path)
            self.assertTrue(save_path.exists())

            loaded_config = TrainingExperimentConfig.load(save_path)
            self.assertEqual(loaded_config.num_train_epochs, 5)
            self.assertEqual(loaded_config.learning_rate, 3e-5)
            self.assertEqual(loaded_config.seed, 123)
            self.assertEqual(loaded_config.model_name_or_path, "Salesforce/codet5-small")

    def test_hardware_detection(self):
        hw = detect_hardware()
        self.assertIn("platform", hw)
        self.assertIn("cuda_available", hw)
        self.assertIn("device_name", hw)
        self.assertIn("recommended_batch_size", hw)
        self.assertIn("recommended_grad_accum", hw)
        self.assertIsInstance(hw["cuda_available"], bool)


if __name__ == "__main__":
    unittest.main()
