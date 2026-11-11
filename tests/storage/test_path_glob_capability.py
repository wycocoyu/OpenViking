# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Remote glob selects the current account backend by capability, not name."""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock


class PathGlobCapabilityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from openviking.server.identity import RequestContext, Role
        from openviking.storage.viking_fs import VikingFS
        from openviking.storage.viking_vector_index_backend import _SingleAccountBackend
        from openviking_cli.session.user_id import UserIdentifier

        self.fs = object.__new__(VikingFS)
        self.fs.glob_config = SimpleNamespace(engine="auto", switch_to_remote_threshold=100)
        self.backend = object.__new__(_SingleAccountBackend)
        self.backend._adapter = SimpleNamespace(mode="custom_backend", supports_path_glob=True)
        self.backend._mode = "custom_backend"
        self.ctx = RequestContext(user=UserIdentifier("account-1", "test-user"), role=Role.ROOT)
        self.store = SimpleNamespace(
            _mode="local",  # Default/store mode must not override the account selection.
            get_account_backend=AsyncMock(return_value=self.backend),
            get_collection_meta=AsyncMock(return_value={
                "Fields": [{"FieldName": "uri", "FieldType": "path"}],
            }),
        )
        self.fs._get_vector_store = lambda: self.store
        self.fs._get_cached_count = AsyncMock(return_value=100)

    async def eligible(self):
        return await self.fs._should_use_vikingdb_glob(
            pattern="docs/**/*.md", uri="viking://resources", node_limit=5, ctx=self.ctx)

    async def test_custom_current_account_backend_is_accepted(self):
        self.assertTrue(await self.eligible())
        self.store.get_account_backend.assert_awaited_once_with("account-1")
        self.fs._get_cached_count.assert_awaited_once_with("viking://resources/docs", self.ctx)
        self.assertEqual(self.backend._mode, "custom_backend")

    async def test_capability_is_not_inferred_from_backend_name(self):
        self.backend._mode = "vikingdb"
        self.backend._adapter.supports_path_glob = False
        self.assertFalse(await self.eligible())
        del self.backend._adapter.supports_path_glob
        self.assertFalse(await self.eligible())
        self.store.get_collection_meta.assert_not_awaited()

    async def test_account_selection_changes_the_capability(self):
        self.assertTrue(await self.eligible())
        self.backend._adapter = SimpleNamespace(mode="another_backend", supports_path_glob=False)
        self.assertFalse(await self.eligible())

    async def test_fs_and_zero_threshold_keep_their_behavior(self):
        self.fs.glob_config.engine = "fs"
        self.assertFalse(await self.eligible())
        self.store.get_account_backend.assert_not_awaited()
        self.fs.glob_config.engine = "auto"
        self.fs.glob_config.switch_to_remote_threshold = 0
        self.assertTrue(await self.eligible())
        self.fs._get_cached_count.assert_not_awaited()

    async def test_path_schema_and_scope_count_are_required(self):
        self.fs._get_cached_count.return_value = 99
        self.assertFalse(await self.eligible())
        for fields in [[], [{"FieldName": "uri"}], [{"FieldName": "uri", "FieldType": "string"}]]:
            self.store.get_collection_meta.return_value = {"Fields": fields}
            with self.subTest(fields=fields):
                self.assertFalse(await self.eligible())

    async def test_backend_metadata_and_count_errors_fall_back(self):
        for method in [self.store.get_account_backend, self.store.get_collection_meta,
                       self.fs._get_cached_count]:
            method.side_effect = RuntimeError("unavailable")
            self.assertFalse(await self.eligible())
            method.side_effect = None

    async def test_builtin_adapters_opt_in_and_base_defaults_to_false(self):
        from openviking.storage.vectordb_adapters.base import CollectionAdapter
        from openviking.storage.vectordb_adapters.vikingdb_private_adapter import (
            VikingDBPrivateCollectionAdapter,
        )
        from openviking.storage.vectordb_adapters.volcengine_adapter import (
            VolcengineCollectionAdapter,
        )

        self.assertFalse(CollectionAdapter.supports_path_glob)
        self.assertTrue(VikingDBPrivateCollectionAdapter.supports_path_glob)
        self.assertTrue(VolcengineCollectionAdapter.supports_path_glob)


if __name__ == "__main__":
    unittest.main()
