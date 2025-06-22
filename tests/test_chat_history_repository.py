import os
import pytest
from unittest import mock

import chat_history_repository as chr


def test_init_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("SQL_DATABASE_URI_LIVE_WRITE", raising=False)
    with pytest.raises(RuntimeError):
        chr.ChatHistoryRepository()


def test_init_uses_env(monkeypatch):
    monkeypatch.setenv("SQL_DATABASE_URI_LIVE_WRITE", "sqlite:///:memory:")
    with mock.patch("chat_history_repository.create_engine") as ce:
        repo = chr.ChatHistoryRepository()
        ce.assert_called_with("sqlite:///:memory:")

