config_dict = dict(

    Acess_config = dict(
        authorization = "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",
    ),

    Model_config = dict(
        model_name = "gpt-3.5-turbo-1106",
        request_address = "https://api.openai.com/v1/chat/completions",
    ),

    Context_manage_config = dict(
        max_context = 3200,
        del_config = dict(
        distance_weights=0.05,
        length_weights=0.4,
        role_weights=1,
        sys_role_ratio=3,
        del_ratio = 0.4,
        max_keep_turns=30)
    ),

    generate_config = dict(
        use_cotomize_param = True,
        param_dict = dict(
        temperature = 1,
        stream = False
        )
    ),
)
