"""
Regression coverage for api/main.py::warm_models' tool-registration wiring
(Phase 8 live pentest findings):

1. `core.real_tools.register_real_tools` -- the function that wires up the
   only REAL (non-sandboxed) tool this project ships, `http.get` -- was
   fully built and tested but never actually called anywhere in the app.
   REGISTER_REAL_TOOLS (new setting, default False) plus this call is the
   fix.

2. A separate, pre-existing bug in the SAME function: tool registration
   used to live after an early `return` gated on WARM_MODELS_ON_STARTUP,
   so a deployment with model warm-up disabled silently never registered
   ANY tools either, even with REGISTER_DEMO_TOOLS explicitly set to True.
   Neither branch of this wiring had ever been directly exercised by a
   test before -- only the standalone register_demo_tools()/
   register_real_tools() functions were tested in isolation, which is
   exactly how a "nothing ever calls this" gap goes unnoticed.

These tests call warm_models() directly (it's a plain async function, not
something that requires a real TestClient lifecycle) with WARM_MODELS_ON_
STARTUP forced False so no real ML models load.
"""
import asyncio
from unittest.mock import patch

import pytest

import api.main as main_mod
from core.tools import get_tool_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Tool registration is a shared, module-level singleton -- reset it
    around each test so these tests can't see each other's registrations,
    and so a real http.get accidentally left registered by a prior live
    testing session doesn't mask a test that expects it to be absent."""
    reg = get_tool_registry()
    original = dict(reg._tools) if hasattr(reg, "_tools") else None
    yield
    if original is not None:
        reg._tools.clear()
        reg._tools.update(original)


def test_register_real_tools_is_never_called_when_flag_is_false():
    with patch("api.main.settings.WARM_MODELS_ON_STARTUP", False), \
         patch("api.main.settings.REGISTER_DEMO_TOOLS", False), \
         patch("api.main.settings.REGISTER_REAL_TOOLS", False), \
         patch("core.real_tools.register_real_tools") as mock_register:
        asyncio.run(main_mod.warm_models())
        mock_register.assert_not_called()


def test_register_real_tools_is_called_when_flag_is_true():
    with patch("api.main.settings.WARM_MODELS_ON_STARTUP", False), \
         patch("api.main.settings.REGISTER_DEMO_TOOLS", False), \
         patch("api.main.settings.REGISTER_REAL_TOOLS", True), \
         patch("core.real_tools.register_real_tools") as mock_register:
        asyncio.run(main_mod.warm_models())
        mock_register.assert_called_once()


def test_register_demo_tools_is_called_when_flag_is_true():
    with patch("api.main.settings.WARM_MODELS_ON_STARTUP", False), \
         patch("api.main.settings.REGISTER_DEMO_TOOLS", True), \
         patch("api.main.settings.REGISTER_REAL_TOOLS", False), \
         patch("core.demo_tools.register_demo_tools") as mock_register:
        asyncio.run(main_mod.warm_models())
        mock_register.assert_called_once()


def test_tool_registration_still_runs_when_warm_models_on_startup_is_false():
    """THE regression this file exists for: registration used to be
    unreachable dead code whenever warm-up was disabled. Both flags on,
    warm-up off -- both registrations must still fire."""
    with patch("api.main.settings.WARM_MODELS_ON_STARTUP", False), \
         patch("api.main.settings.REGISTER_DEMO_TOOLS", True), \
         patch("api.main.settings.REGISTER_REAL_TOOLS", True), \
         patch("core.demo_tools.register_demo_tools") as mock_demo, \
         patch("core.real_tools.register_real_tools") as mock_real:
        asyncio.run(main_mod.warm_models())
        mock_demo.assert_called_once()
        mock_real.assert_called_once()


def test_http_get_is_actually_reachable_end_to_end_once_registered():
    """Not just 'was the function called' -- prove the real consequence:
    after warm_models() runs with REGISTER_REAL_TOOLS=True, http.get is
    genuinely present in the shared tool registry POST /api/v1/tools/call
    reads from."""
    with patch("api.main.settings.WARM_MODELS_ON_STARTUP", False), \
         patch("api.main.settings.REGISTER_DEMO_TOOLS", False), \
         patch("api.main.settings.REGISTER_REAL_TOOLS", True):
        asyncio.run(main_mod.warm_models())

    reg = get_tool_registry()
    names = [spec.name for spec in reg.list_tools()]
    assert "http.get" in names
