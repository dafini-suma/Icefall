#!/usr/bin/env python

"""Abstract class for data parser."""

from abc import ABC, abstractmethod


class DataParser(ABC):
    @abstractmethod
    def parse(self):
        pass
