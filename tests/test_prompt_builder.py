from agent.core.prompt_builder import PromptBuilder


def test_custom_table_info_uses_custom_info():
    builder = PromptBuilder()
    info = builder.build_custom_table_info(["sales_order"], db="live")
    assert "sales_order" in info
    # customInfo should override default formatting
    assert "links" in info["sales_order"]


def test_system_prompt_includes_custom_info():
    builder = PromptBuilder()
    prompt = builder.build_system_prompt(
        db="common", allowed_tables=["customer_wallet"]
    )
    assert "customer_id joins" in prompt
