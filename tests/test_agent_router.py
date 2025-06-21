import importlib.util
import sys
import types
from pathlib import Path


def _load_module():
    packages = {
        "langchain": {},
        "langchain.agents": {
            "create_openai_functions_agent": lambda *a, **k: None,
            "AgentExecutor": object,
        },
        "langchain_core.prompts": {
            "ChatPromptTemplate": object,
            "MessagesPlaceholder": object,
        },
        "langchain_core.output_parsers": {"StrOutputParser": object},
        "langchain_core.runnables": {
            "RunnablePassthrough": object,
            "RunnableLambda": object,
            "RunnableBranch": object,
        },
        "langchain_openai": {"ChatOpenAI": object},
    }

    for name, attrs in packages.items():
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        for attr, val in attrs.items():
            setattr(mod, attr, val)

    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    sql_tool = types.ModuleType("tools.sql_tool")
    sql_tool.get_live_sql_tools = lambda: None
    sql_tool.get_common_sql_tools = lambda: None
    sys.modules["tools.sql_tool"] = sql_tool

    spec = importlib.util.spec_from_file_location(
        "agent_router", Path(__file__).resolve().parents[1] / "agent_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent_router = _load_module()


def test_extract_customer_id_negative():
    query = "What is the status of order 78910?"
    assert agent_router._extract_customer_id(query) is None
