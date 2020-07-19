# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import tokenization
import six
import tensorflow as tf


class TokenizationTest(tf.test.TestCase):

  def test_full_tokenizer(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing", ",", "1@1", "0@0", "km", "1@3", "2@2", "3@1"
    ]
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as vocab_writer:
      vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
      vocab_file = vocab_writer.name

    tokenizer = tokenization.FullTokenizer(vocab_file)
    os.unlink(vocab_file)

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running")
    self.assertAllEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
    self.assertAllEqual(
        tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9])

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running for 10 km")
    self.assertAllEqual(tokens,
                        ['un', '##want', '##ed', ',', 'runn', '##ing', '[UNK]', '1@1', '0@0', 'km'])
    self.assertAllEqual(
        tokenizer.convert_tokens_to_ids(tokens), [7, 4, 5, 10, 8, 9, 0, 11, 12, 13])

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running for 10km")
    self.assertAllEqual(tokens,
                        ['un', '##want', '##ed', ',', 'runn', '##ing', '[UNK]', 
                         '1@1', '0@0', 'km'])
    self.assertAllEqual(
        tokenizer.convert_tokens_to_ids(tokens),
        [7, 4, 5, 10, 8, 9, 0, 11, 12, 13])

    tokens = tokenizer.tokenize(u"UNwant\u00E9d,running for 1,230km")
    self.assertAllEqual(tokens,
                        ['un', '##want', '##ed', ',', 'runn', '##ing', '[UNK]',
                         '1@3', ',', '2@2', '3@1', '0@0', 'km'])
    self.assertAllEqual(
        tokenizer.convert_tokens_to_ids(tokens),
        [7, 4, 5, 10, 8, 9, 0, 14, 10, 15, 16, 12, 13])

  def test_chinese(self):
    tokenizer = tokenization.BasicTokenizer()

    self.assertAllEqual(
        tokenizer.tokenize(u"ah\u535A\u63A8zz"),
        [u"ah", u"\u535A", u"\u63A8", u"zz"])

  def test_basic_tokenizer_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["hello", "!", "how", "are", "you", "?"])
    self.assertAllEqual(tokenizer.tokenize(u"H\u00E9llo"), ["hello"])

  def test_basic_tokenizer_no_lower(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertAllEqual(
        tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  "),
        ["HeLLo", "!", "how", "Are", "yoU", "?"])

  def test_run_split_on_punc(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertAllEqual(
        tokenizer._run_split_on_punc("How are you? I am fine. Thank you!"),
        ["How are you", "?", " I am fine", ".", " Thank you", "!"])

  def test_run_split_on_num(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertEqual(
        tokenizer._run_split_on_num("10km"),
        (["10", "km"], [True, False]))

    self.assertEqual(
        tokenizer._run_split_on_num("12,345.67km"),
        (["12,345.67", "km"], [True, False]))

    self.assertEqual(
        tokenizer._run_split_on_num("My10modelshavebeentrainingfor20days"),
        (["My", "10", "modelshavebeentrainingfor", "20", "days"],
        [False, True, False, True, False]))

    self.assertEqual(
        tokenizer._run_split_on_num("Ittakes10,000tobeanexpert."),
        (["Ittakes", "10,000", "tobeanexpert."],
        [False, True, False]))

    self.assertEqual(
        tokenizer._run_split_on_num("Ittakes10000tobeanexpert."),
        (["Ittakes", "10000", "tobeanexpert."],
        [False, True, False]))

  def test_basic_tokenizer_with_number(self):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=False)

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1,234.5 meter"),
        ["unwanted", "running", "for", "1,234.5", "meter"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1234.5 meter"),
        ["unwanted", "running", "for", "1234.5", "meter"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1234 meter"),
        ["unwanted", "running", "for", "1234", "meter"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1,234.5meter"),
        ["unwanted", "running", "for", "1,234.5", "meter"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1234.5meter"),
        ["unwanted", "running", "for", "1234.5", "meter"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1234meter"),
        ["unwanted", "running", "for", "1234", "meter"])

  def test_wordpiece_tokenizer(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing", '1@3', '2@2', '4@0', '.', '5@-1', "1@5", "2@4", "3@3", "4@2",
        "5@1", "6@0", "7@-1", "doll", "##ars"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
      vocab[token] = i
    tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

    self.assertAllEqual(tokenizer.tokenize(""), [])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running"),
        ["un", "##want", "##ed", "runn", "##ing"])

    self.assertAllEqual(
        tokenizer.tokenize("unwantedX running"), ["[UNK]", "runn", "##ing"])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 1,234.5 meter"),
        ['un', '##want', '##ed', 'runn', '##ing', '[UNK]', 
         '1@3', '[UNK]', '2@2', '[UNK]', '4@0', '.', '5@-1', '[UNK]'])

    self.assertAllEqual(
        tokenizer.tokenize("unwanted running for 123456.7 dollars"),
        ['un', '##want', '##ed', 'runn', '##ing', '[UNK]',
         "1@5", "2@4", "3@3", "4@2", "5@1", "6@0", '.', "7@-1",
         "doll", "##ars"])

  def test_convert_by_vocab(self):
    vocab_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn",
        "##ing"
    ]

    vocab = {}
    for (i, token) in enumerate(vocab_tokens):
      vocab[token] = i

    self.assertAllEqual(
        tokenization.convert_by_vocab(
            vocab, ["un", "##want", "##ed", "runn", "##ing"]), [7, 4, 5, 8, 9])

  def test_is_whitespace(self):
    self.assertTrue(tokenization._is_whitespace(u" "))
    self.assertTrue(tokenization._is_whitespace(u"\t"))
    self.assertTrue(tokenization._is_whitespace(u"\r"))
    self.assertTrue(tokenization._is_whitespace(u"\n"))
    self.assertTrue(tokenization._is_whitespace(u"\u00A0"))

    self.assertFalse(tokenization._is_whitespace(u"A"))
    self.assertFalse(tokenization._is_whitespace(u"-"))

  def test_is_control(self):
    self.assertTrue(tokenization._is_control(u"\u0005"))

    self.assertFalse(tokenization._is_control(u"A"))
    self.assertFalse(tokenization._is_control(u" "))
    self.assertFalse(tokenization._is_control(u"\t"))
    self.assertFalse(tokenization._is_control(u"\r"))
    self.assertFalse(tokenization._is_control(u"\U0001F4A9"))

  def test_is_punctuation(self):
    self.assertTrue(tokenization._is_punctuation(u"-"))
    self.assertTrue(tokenization._is_punctuation(u"$"))
    self.assertTrue(tokenization._is_punctuation(u"`"))
    self.assertTrue(tokenization._is_punctuation(u"."))

    self.assertFalse(tokenization._is_punctuation(u"A"))
    self.assertFalse(tokenization._is_punctuation(u" "))

  def test_is_number(self):
    self.assertTrue(tokenization._is_number('1,000'))
    self.assertTrue(tokenization._is_number('1000'))
    self.assertTrue(tokenization._is_number('1000.12'))
    self.assertFalse(tokenization._is_number('1a2b'))
    self.assertFalse(tokenization._is_number('bert'))

  def test_tokenize_number_with_position(self):
    self.assertEqual(tokenization.tokenize_number_with_position(
                     ['1', ',', '0', '0', '0', '.', '2', '3'],
                     {'1@3': 0, ',': 1, '0@1': 3, '0@0': 4, '.': 5, '2@-1': 6, '3@-2': 7},
                     '[UNK]'),
                     ['1@3', ',', '[UNK]', '0@1', '0@0', '.', '2@-1', '3@-2'])
    self.assertEqual(tokenization.tokenize_number_with_position(
                     ['1', ',', '0', '0', '0'],
                     {'1@3': 0, ',': 1, '0@2': 2, '0@1': 3, '0@0': 4},
                     '[UNK]'),
                     ['1@3', ',', '0@2', '0@1', '0@0'])
    self.assertEqual(tokenization.tokenize_number_with_position(
                     ['1'],
                     {'1@0': 0},
                     '[UNK]'),
                     ['1@0'])


if __name__ == "__main__":
  tf.test.main()
