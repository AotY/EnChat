# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import io


class StopWord:

    def __init__(self, file_path):
        self.file_path = file_path
        self.load_stop_words()
        self.stop_words = []

    def load_stop_words(self):
        with io.open(self.file_path, 'r', encoding='utf-8') as fi:
            for line in fi:
                line = line.strip()
                # Skip blank lines
                if not line:
                    continue
                if line not in self.stop_words:
                    self.stop_words.append(line)

    def remove_words(self, line):
        cleaned_words = [token for token in line.split() if token not in self.stop_words]
        return cleaned_words
