# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
import unittest

from sglang_omni.models.qwen3_tts.engine_builder import (
    _talker_requires_speech_tokenizer,
)


class TestQwen3TtsTalkerTokenizerPolicy(unittest.TestCase):
    def test_base_keeps_speech_tokenizer(self):
        self.assertTrue(
            _talker_requires_speech_tokenizer(SimpleNamespace(tts_model_type="base"))
        )

    def test_custom_voice_skips_speech_tokenizer(self):
        for model_type in ("custom_voice", "customvoice", "custom-voice"):
            with self.subTest(model_type=model_type):
                self.assertFalse(
                    _talker_requires_speech_tokenizer(
                        SimpleNamespace(tts_model_type=model_type)
                    )
                )

    def test_voice_design_skips_speech_tokenizer(self):
        for model_type in ("voice_design", "voicedesign", "voice-design"):
            with self.subTest(model_type=model_type):
                self.assertFalse(
                    _talker_requires_speech_tokenizer(
                        SimpleNamespace(tts_model_type=model_type)
                    )
                )

    def test_unknown_model_type_keeps_historical_behavior(self):
        self.assertTrue(
            _talker_requires_speech_tokenizer(
                SimpleNamespace(tts_model_type="future_variant")
            )
        )

    def test_missing_model_type_keeps_historical_behavior(self):
        self.assertTrue(_talker_requires_speech_tokenizer(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
